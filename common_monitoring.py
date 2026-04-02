from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional
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
        self.latest_risk = {"risk_text": None, "updated_at": None, "source_name": None, "analysis": None, "detections": None}
        self.camera_state_lock = threading.Lock()
        self.camera_state = {
            "registered": False, "ip_address": None, "camera_id": None, "camera_pw": None,
            "rtsp_port": 554, "rtsp_path": "/stream1", "last_health": None
        }
        self.video_state_lock = threading.Lock()
        self.video_state = {"running": False, "video_name": None, "last_frame_path": None, "last_frame_updated_at": None, "last_error": None}
        self.video_worker_thread: Optional[threading.Thread] = None
        self.video_stop_event = threading.Event()
        self.yolo_model: Optional[YOLO] = None
        self.qwen_runner: Optional[tensorrt.TensorRTQwenRunner] = None


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
        cv2.putText(output, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return output


def draw_status(frame, analysis):
    output = frame.copy()
    text = f"person: {analysis['person_count']}  fire: {'yes' if analysis['has_fire'] else 'no'}  smoke: {'yes' if analysis['has_smoke'] else 'no'}"
    color = (0, 0, 255) if (analysis["has_fire"] or analysis["has_smoke"]) else (255, 255, 255)
    cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
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
    return {"person_count": person_count, "has_fire": has_fire, "has_smoke": has_smoke}


def build_vlm_prompt(detections, analysis):
    detection_lines = []
    for i, det in enumerate(detections):
        detection_lines.append(f"[{i}] class={det['class_name']} conf={det['conf']:.2f} box=({int(det['x1'])}, {int(det['y1'])}, {int(det['x2'])}, {int(det['y2'])})")
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


def run_single_frame_analysis(frame, source_name: Optional[str] = None):
    if state.yolo_model is None or state.qwen_runner is None:
        raise RuntimeError("모델이 아직 로드되지 않았습니다.")
    resize_frame = resize_to_640(frame)
    detections = yolo_detection.detect_positions_with_class_on_frame(state.yolo_model, resize_frame)
    analysis = analyze_detected_classes(detections)
    display_frame = draw_detections(resize_frame, detections)
    display_frame = draw_status(display_frame, analysis)
    update_last_frame(display_frame)
    risk_text = None
    if analysis["has_fire"] or analysis["has_smoke"]:
        rgb_frame = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        prompt = build_vlm_prompt(detections, analysis)
        risk_text = state.qwen_runner.infer(image_input=pil_image, user_text=prompt, max_new_tokens=64)
        update_latest_risk(risk_text, source_name, analysis, detections)
    return {"detections": detections, "analysis": analysis, "risk_text": risk_text, "frame_path": str(LAST_FRAME_PATH)}


def build_rtsp_url(ip_address: str, camera_id: str, camera_pw: str, rtsp_port: int = 554, rtsp_path: str = "/stream1") -> str:
    user = quote(camera_id, safe="")
    password = quote(camera_pw, safe="")
    path = rtsp_path if rtsp_path.startswith("/") else f"/{rtsp_path}"
    return f"rtsp://{user}:{password}@{ip_address}:{rtsp_port}{path}"


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
    return b"--frame\r\n" + b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
