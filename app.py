import io
from threading import Thread
from typing import Dict, Optional

import torch
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    TextIteratorStreamer,
)
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
    당신은 건설현장 안전 알림 문장을 생성하는 시스템입니다.

    입력 요약:
    {scene_summary_text}

    반드시 한국어로만, 반드시 아래 한 줄 형식으로만 답하세요.

    형식:
    알림 이유: <이유> / 장면 설명: <설명>

    규칙:
    - 반드시 한 줄만 출력하세요.
    - "알림 이유:"로 시작하고, 뒤에 "/ 장면 설명:"을 붙이세요.
    - 안전모 미착용 인원 수와 조끼 미착용 인원 수가 있으면 이유에 포함하세요.
    - 화재, 낙상, 분진/연기, 뚜렷한 위험 상황이 보이면 그것을 이유에서 우선 언급하세요.
    - 장면 설명에는 화면에 보이는 작업 상황과 주변 환경만 짧게 쓰세요.
    - 영어는 절대 쓰지 마세요.
    - 좌표, confidence, class id, bounding box, 추론 과정은 절대 쓰지 마세요.

    예시:
    알림 이유: 안전모를 착용하지 않은 작업자 1명이 있어 알림이 발생했습니다. / 장면 설명: 화면에는 작업자들이 현장에서 작업 중이며 주변에 장비와 자재가 보입니다.
    """.strip()

    if user_prompt and user_prompt.strip():
        prompt += f"\n추가 지시사항:\n{user_prompt.strip()}"

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

        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            streamer=streamer,
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        def stream_text():
            try:
                for chunk in streamer:
                    # chunk를 바로 흘려보냄
                    yield chunk
            except Exception:
                yield ""

        return StreamingResponse(
            stream_text(),
            media_type="text/plain; charset=utf-8",
        )

    except torch.cuda.OutOfMemoryError:
        raise HTTPException(status_code=500, detail="GPU 메모리 부족(OOM)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {e}")