import io
from typing import Dict, Optional

import torch
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct-FP8"
TARGET_SIZE = 640
MAX_PIXELS = TARGET_SIZE * TARGET_SIZE

app = FastAPI(title="Qwen3-VL Scene-Summary Server")

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


def resize_force_square(image: Image.Image, size: int = TARGET_SIZE) -> Image.Image:
    return image.resize((size, size))


def parse_scene_summary(raw_text: str) -> Dict[str, str]:
    result: Dict[str, str] = {}

    for line in raw_text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue

        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()

    return result


def to_int(summary: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(summary.get(key, default))
    except (TypeError, ValueError):
        return default


def normalize_scene_summary(summary: Dict[str, str]) -> str:
    worker_count = to_int(summary, "workers", 0)
    no_helmet_count = to_int(summary, "workers_without_helmet", 0)
    no_vest_count = to_int(summary, "workers_without_vest", 0)
    has_any_risk = summary.get("has_any_risk", "no")

    return "\n".join([
        f"workers={worker_count}",
        f"workers_without_helmet={no_helmet_count}",
        f"workers_without_vest={no_vest_count}",
        f"has_any_risk={has_any_risk}",
    ])


def build_prompt(user_prompt: Optional[str], scene_summary_text: str) -> str:
    prompt = f"""
Korean construction safety alert.

Summary:
{scene_summary_text}

Write 1-2 short Korean sentences.
Sentence 1: alert reason.
Include counts of workers without helmets or vests if applicable.
If visible, prioritize fire, fall, dust/smoke, or obvious danger.
Sentence 2: brief scene/environment description.
No coordinates, confidence, class IDs, boxes, or reasoning.
""".strip()

    if user_prompt and user_prompt.strip():
        prompt += f"\n{user_prompt.strip()}"

    return prompt

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "target_size": TARGET_SIZE,
        "max_pixels": MAX_PIXELS,
    }


@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    detections: str = Form(...),
    prompt: str = Form(""),
):
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 로드 실패: {e}")

    image = resize_force_square(image, TARGET_SIZE)

    parsed_summary = parse_scene_summary(detections)
    scene_summary_text = normalize_scene_summary(parsed_summary)
    final_prompt = build_prompt(prompt, scene_summary_text)

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
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]

        result_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return JSONResponse(content={"result": result_text})

    except torch.cuda.OutOfMemoryError:
        raise HTTPException(status_code=500, detail="GPU 메모리 부족(OOM)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {e}")