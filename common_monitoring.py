from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from urllib import request as urllib_request
from urllib.parse import quote

import cv2
from PIL import Image
from fastapi import HTTPException
from ultralytics import YOLO

import qwen_tensorrt as tensorrt
import yolo_detection


BASE_DIR = Path.home() / "detection_VLM"
VIDEO_DIR = BASE_DIR
RUNTIME_DIR = Path.home() / "edgellm_work" / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
LAST_FRAME_PATH = RUNTIME_DIR / "latest_frame.jpg"


class SharedState:
    def __init__(self):
        self.model_status = {"model": "loading"}
        self.model_lock = threading.Lock()

        self.latest_risk_lock = threading.Lock()
        self.latest_risk = {
            "risk_text": None,
            "updated_at": None,
            "source_name": None,
            "analysis": None,
            "detections": None,
        }

        self.video_state_lock = threading.Lock()
        self.video_state = {
            "running": False,
            "video_name": None,
            "last_frame_path": None,
            "last_frame_updated_at": None,
            "last_error": None,
        }

        self.video_worker_thread: Optional[threading.Thread] = None
        self.video_stop_event = threading.Event()

        self.yolo_model: Optional[YOLO] = None
        self.qwen_runner: Optional[tensorrt.TensorRTQwenRunner] = None

        self.internal_vlm_analysis_lock = threading.Lock()
        self.internal_vlm_analysis = None
        self.internal_vlm_callback_url = "http://127.0.0.1:8000/api/internal/vlm-analysis"

        self.stream_frame_lock = threading.Lock()
        self.latest_stream_frame = None

        self.vlm_cooldown_lock = threading.Lock()
        self.vlm_cooldown_seconds = 30.0
        self.last_vlm_trigger_by_source = {}


state = SharedState()


def startup_models():
    try:
        state.yolo_model = YOLO(str(BASE_DIR / "best.engine"))
        state.qwen_runner = tensorrt.TensorRTQwenRunner(
            engine_dir="~/edgellm_work/engines/qwen3-vl-2b",
            multimodal_engine_dir="~/edgellm_work/visual_engines/qwen3-vl-2b",
            llm_inference_bin="~/TensorRT-Edge-LLM/build/examples/llm/llm_inference",
            plugin_path="~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so",
            work_dir="~/edgellm_work/runtime",
        )
        with state.model_lock:
            state.model_status["model"] = "ok"
    except Exception as e:
        with state.model_lock:
            state.model_status["model"] = f"error: {e}"


def ensure_models_ready():
    if state.model_status.get("model") != "ok":
        raise HTTPException(status_code=503, detail="모델이 아직 준비되지 않았습니다.")


def resize_to_640(frame):
    return cv2.resize(frame, (640, 640))


def draw_detections(frame, detections):
    output = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = map(int, (det["x1"], det["y1"], det["x2"], det["y2"]))
        class_name = det["class_name"]
        conf = det["conf"]

        label = f"{class_name} {conf:.2f}"

        color = (0, 255, 0)
        if class_name.lower() == "fire":
            color = (0, 0, 255)
        elif class_name.lower() == "smoke":
            color = (0, 165, 255)
        elif class_name.lower() == "person":
            color = (255, 0, 0)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return output


def draw_status(frame, analysis):
    output = frame.copy()

    text = (
        f"person: {analysis['person_count']}  "
        f"fire: {'yes' if analysis['has_fire'] else 'no'}  "
        f"smoke: {'yes' if analysis['has_smoke'] else 'no'}"
    )

    color = (0, 0, 255) if (analysis["has_fire"] or analysis["has_smoke"]) else (255, 255, 255)

    cv2.putText(
        output,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    return output


def analyze_detected_classes(detections):
    person_count = 0
    has_fire = False
    has_smoke = False

    for det in detections:
        class_name = det.get("class_name", "").lower()
        if class_name == "person":
            person_count += 1
        elif class_name == "fire":
            has_fire = True
        elif class_name == "smoke":
            has_smoke = True

    return {
        "person_count": person_count,
        "has_fire": has_fire,
        "has_smoke": has_smoke,
    }


def build_vlm_prompt(detections, analysis):
    detection_lines = []
    for i, det in enumerate(detections):
        detection_lines.append(
            f"[{i}] class={det['class_name']} conf={det['conf']:.2f} "
            f"box=({int(det['x1'])}, {int(det['y1'])}, {int(det['x2'])}, {int(det['y2'])})"
        )

    detection_text = "\n".join(detection_lines) if detection_lines else "탐지 결과 없음"

    return f"""당신은 건설 현장 안전 모니터링 도우미입니다.
반드시 한국어로만 답변하세요.

탐지 요약:
- 사람 수: {analysis['person_count']}
- 화재 여부: {'예' if analysis['has_fire'] else '아니오'}
- 연기 여부: {'예' if analysis['has_smoke'] else '아니오'}

YOLO 탐지 결과:
{detection_text}

출력 형식:
위험상황: <짧게 1~2문장>

규칙:
- 이미지를 가장 우선해서 판단하세요.
- YOLO 탐지 결과는 보조 정보로만 사용하세요.
- 위험한 상황이면 무엇이 위험한지 짧고 분명하게 설명하세요.
- 위험하지 않으면 현재 상황을 짧게 설명하세요.
- 불꽃이나 연기가 보이면 반드시 언급하세요.
- 같은 내용을 반복하지 마세요.
- 길게 쓰지 마세요.
""".strip()


def update_latest_risk(risk_text, source_name, analysis, detections):
    with state.latest_risk_lock:
        state.latest_risk["risk_text"] = risk_text
        state.latest_risk["updated_at"] = int(time.time() * 1000)
        state.latest_risk["source_name"] = source_name
        state.latest_risk["analysis"] = analysis
        state.latest_risk["detections"] = detections


def update_last_frame(display_frame):
    cv2.imwrite(str(LAST_FRAME_PATH), display_frame)
    with state.video_state_lock:
        state.video_state["last_frame_path"] = str(LAST_FRAME_PATH)
        state.video_state["last_frame_updated_at"] = int(time.time() * 1000)


def update_latest_stream_frame(display_frame):
    with state.stream_frame_lock:
        state.latest_stream_frame = display_frame.copy()


def can_run_vlm(source_name: Optional[str]) -> bool:
    if not source_name:
        return True

    now = time.time()
    with state.vlm_cooldown_lock:
        last_ts = state.last_vlm_trigger_by_source.get(source_name)
        if last_ts is None:
            return True
        return (now - last_ts) >= state.vlm_cooldown_seconds


def mark_vlm_trigger(source_name: Optional[str]) -> None:
    if not source_name:
        return

    with state.vlm_cooldown_lock:
        state.last_vlm_trigger_by_source[source_name] = time.time()


def resolve_ev_code_name(analysis, detections) -> str:
    if analysis.get("has_fire"):
        return "FIRE_DETECTED"
    if analysis.get("has_smoke"):
        return "SMOKE_DETECTED"
    return "RISK_DETECTED"


def now_kst_iso() -> str:
    kst = timezone(timedelta(hours=9))
    return datetime.now(kst).isoformat(timespec="seconds")


def post_internal_vlm_analysis(ip_address: str, ev_code_name: str, risk_text: str, event_time: str):
    payload = {
        "ip_address": ip_address,
        "ev_code_name": ev_code_name,
        "risk_text": risk_text,
        "time": event_time,
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib_request.Request(
        state.internal_vlm_callback_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib_request.urlopen(req, timeout=3) as resp:
        body = resp.read().decode("utf-8")

    with state.internal_vlm_analysis_lock:
        state.internal_vlm_analysis = payload

    return body


def send_internal_vlm_if_needed(source_name: str, analysis, detections, risk_text: str):
    if not risk_text:
        return

    ev_code_name = resolve_ev_code_name(analysis, detections)
    event_time = now_kst_iso()

    try:
        post_internal_vlm_analysis(
            ip_address=source_name,
            ev_code_name=ev_code_name,
            risk_text=risk_text,
            event_time=event_time,
        )
    except Exception as e:
        print(f"[internal vlm-analysis 전송 실패] {e}", flush=True)


def run_single_frame_analysis(frame, source_name: Optional[str] = None):
    if state.yolo_model is None or state.qwen_runner is None:
        raise RuntimeError("모델이 아직 로드되지 않았습니다.")

    resize_frame = resize_to_640(frame)
    detections = yolo_detection.detect_positions_with_class_on_frame(state.yolo_model, resize_frame)
    analysis = analyze_detected_classes(detections)

    display_frame = draw_detections(resize_frame, detections)
    display_frame = draw_status(display_frame, analysis)

    update_last_frame(display_frame)
    update_latest_stream_frame(display_frame)

    risk_text = None
    if analysis["has_fire"]:
        if can_run_vlm(source_name):
            rgb_frame = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            prompt = build_vlm_prompt(detections, analysis)

            risk_text = state.qwen_runner.infer(
                image_input=pil_image,
                user_text=prompt,
                max_new_tokens=64,
            )

            mark_vlm_trigger(source_name)
            update_latest_risk(risk_text, source_name, analysis, detections)

            if source_name:
                send_internal_vlm_if_needed(
                    source_name=source_name,
                    analysis=analysis,
                    detections=detections,
                    risk_text=risk_text,
                )

    return {
        "detections": detections,
        "analysis": analysis,
        "risk_text": risk_text,
        "frame_path": str(LAST_FRAME_PATH),
    }


def build_rtsp_url(ip_addressess: str, camera_id: str, camera_pw: str, rtsp_port: int = 554, rtsp_path: str = "/stream1") -> str:
    user = quote(camera_id, safe="")
    password = quote(camera_pw, safe="")
    path = rtsp_path if rtsp_path.startswith("/") else f"/{rtsp_path}"
    return f"rtsp://{user}:{password}@{ip_addressess}:{rtsp_port}{path}"


def test_rtsp_connection(rtsp_url: str) -> bool:
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        cap.release()
        return False

    ret, _ = cap.read()
    cap.release()
    return bool(ret)


def mjpeg_frame_bytes(frame):
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return None

    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" +
        encoded.tobytes() +
        b"\r\n"
    )