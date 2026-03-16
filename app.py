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

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

app = FastAPI(title="Qwen3-VL YOLO-grounded Server")


# -----------------------------
# 모델 로드
# -----------------------------
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
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
def resize_keep_ratio(image: Image.Image, target_size: int = 664) -> Image.Image:
    """
    긴 변 기준이 아니라, 여기서는 사용자가 664x664로 넣는다고 했으므로
    입력이 다르더라도 강제로 664x664로 맞추는 단순 버전.
    YOLO 좌표가 원본 기준이라면 반드시 같은 해상도를 기준으로 넣어야 함.
    """
    return image.resize((target_size, target_size))


def parse_detections_text(raw_text: str, class_map: Dict[int, str]) -> List[Dict[str, Any]]:
    """
    예시 입력:
    [0] class=2 conf=0.799316 box=(540.75, 515.625, 998.25, 984.375)
    """
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
You are analyzing a single scene image for construction safety monitoring.

Use BOTH:
1) the image itself
2) the YOLO detections provided below

YOLO detections are structured hints and should be treated as primary evidence for object presence and approximate location.
Do not invent objects that are not visible or not supported by the detections.
If something is uncertain, say it conservatively.

YOLO detection information:
{detections_summary}

Your task:
- Output only 1 or 2 short lines in Korean.
- Keep the answer concise and practical.
- Focus only on these points:
  1) whether workers are present
  2) whether workers appear to be wearing helmets
  3) whether workers appear to be wearing vests
  4) whether fire or an obvious dangerous situation is visible
- If there are workers without nearby helmet detections, describe them as workers possibly not wearing helmets.
- If there are workers without nearby vest detections, describe them as workers possibly not wearing vests.
- If all visible workers appear equipped properly, say so clearly.
- If fire is visible, mention it first.
- Do not mention coordinates, confidence scores, class IDs, or bounding boxes.
- Do not explain your reasoning.
- Do not use bullet points or numbering.
- Use a natural Korean safety-report style.

Example style:
작업자들이 작업중이며, 안전모를 착용하지 않은 작업자 2명이 보입니다.
작업자들이 안전모와 조끼를 착용하고 작업중입니다.
작업장에 불이 났으며, 작업자는 보이지 않습니다.
"""

    if user_prompt and user_prompt.strip():
        base_prompt += f"\nAdditional instruction from user:\n{user_prompt.strip()}\n"

    return base_prompt.strip()


# -----------------------------
# API
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


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

    # 필요 시 664x664로 강제 리사이즈
    image = resize_keep_ratio(image, 664)

    # 클래스 맵 파싱
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
                max_new_tokens=220,
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
        )[0]

        return JSONResponse(
            content={"result": result_text}
        )

    except torch.cuda.OutOfMemoryError:
        raise HTTPException(status_code=500, detail="GPU 메모리 부족(OOM)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {e}")