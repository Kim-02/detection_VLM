import io
import json
import re
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# FP8 모델 사용
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct-FP8"

# max_pixels는 한 변 길이가 아니라 총 픽셀 수 기준
# 640x640 제한
MAX_PIXELS = 640 * 640

app = FastAPI(title="Qwen3-VL YOLO-grounded Server")


# -----------------------------
# 모델 로드
# -----------------------------
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    max_pixels=MAX_PIXELS,
    trust_remote_code=True,
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)


# -----------------------------
# 기본 클래스 맵
# -----------------------------
DEFAULT_CLASS_MAP = {
    0: "helmet",
    1: "vest",
    2: "person",
    3: "danger_vehicle",
}


# -----------------------------
# 유틸
# -----------------------------
def resize_force_664(image: Image.Image) -> Image.Image:
    return image.resize((664, 664))


def parse_detections_text(raw_text: str, class_map: Dict[int, str]) -> List[Dict[str, Any]]:
    pattern = re.compile(
        r"\[(?P<idx>\d+)\]\s+class=(?P<class_id>\d+)\s+conf=(?P<conf>[0-9.]+)\s+box=\((?P<x1>[0-9.]+),\s*(?P<y1>[0-9.]+),\s*(?P<x2>[0-9.]+),\s*(?P<y2>[0-9.]+)\)"
    )

    detections = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue

        match = pattern.match(line)
        if not match:
            continue

        class_id = int(match.group("class_id"))
        det = {
            "index": int(match.group("idx")),
            "class_id": class_id,
            "class_name": class_map.get(class_id, f"class_{class_id}"),
            "confidence": float(match.group("conf")),
            "box": [
                float(match.group("x1")),
                float(match.group("y1")),
                float(match.group("x2")),
                float(match.group("y2")),
            ],
        }
        detections.append(det)

    return detections


def summarize_detections(detections: List[Dict[str, Any]]) -> str:
    if not detections:
        return "No detections were provided."

    counts: Dict[str, int] = {}
    lines: List[str] = []

    for det in detections:
        name = det["class_name"]
        counts[name] = counts.get(name, 0) + 1

    count_text = ", ".join([f"{k}: {v}" for k, v in sorted(counts.items())])

    lines.append(f"Detected objects summary: {count_text}.")
    lines.append("Detailed detections:")

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        lines.append(
            f"- idx={det['index']}, class={det['class_name']} (id={det['class_id']}), "
            f"conf={det['confidence']:.3f}, box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )

    return "\n".join(lines)


def build_prompt(user_prompt: Optional[str], detections_summary: str) -> str:
    base_prompt = f"""
You are a construction safety monitoring system.

You must use both:
1) the image itself
2) the YOLO detection information below

The YOLO detections are the main evidence for object presence and approximate location.
Do not invent anything that is not supported by the image or the detections.
If something is uncertain, describe it conservatively.

YOLO detection information:
{detections_summary}

Your task:
- Output only 1 or 2 sentences in Korean.
- In the first sentence, explain why this alert was triggered.
- The alert reason may include missing helmet, missing vest, worker presence, dangerous vehicle proximity, or suspected fire.
- In the second sentence, briefly describe what is happening in the scene.
- Mention visible workers, PPE status, and relevant hazards in a short and natural way.
- Do not mention coordinates, confidence scores, class IDs, bounding boxes, or your reasoning process.
- Keep the response short, practical, and report-like.
- If the situation is not fully clear, use cautious expressions such as "appears to be" or "is suspected."
- If there is no clear fire or major hazard, do not mention it unnecessarily.

Example outputs:
A worker without a helmet appears to be present, which triggered this alert. Workers are visible in the scene, and at least one appears to be working without a helmet.
A worker without a vest appears to be present, which triggered this alert. Several workers are visible, and at least one appears to be without a safety vest.
This alert was triggered because a worker is close to a dangerous vehicle. A worker and a vehicle are visible in close proximity in the scene.
This alert was triggered because fire is suspected in the work area. Flames are visible, and no worker is clearly seen.
"""

    if user_prompt and user_prompt.strip():
        base_prompt += f"\nAdditional instruction:\n{user_prompt.strip()}\n"

    return base_prompt.strip()

# -----------------------------
# API
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "max_pixels": MAX_PIXELS}


@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    detections: str = Form(...),
    prompt: str = Form(""),
    class_map_json: str = Form("")
):
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 로드 실패: {e}")

    # 입력 이미지는 664x664로 맞춤
    image = resize_force_664(image)

    class_map = DEFAULT_CLASS_MAP.copy()
    if class_map_json.strip():
        try:
            user_map_raw = json.loads(class_map_json)
            user_map = {int(k): str(v) for k, v in user_map_raw.items()}
            class_map.update(user_map)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"class_map_json 파싱 실패: {e}")

    parsed_detections = parse_detections_text(detections, class_map)
    detections_summary = summarize_detections(parsed_detections)
    final_prompt = build_prompt(prompt, detections_summary)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": final_prompt},
            ],
        }
    ]

    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,   # 220 -> 64로 축소
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]

        result_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        return JSONResponse(content={"result": result_text})

    except torch.cuda.OutOfMemoryError:
        raise HTTPException(status_code=500, detail="GPU 메모리 부족(OOM)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {e}")