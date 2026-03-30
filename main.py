import threading
import queue
import time
import cv2
from PIL import Image

import yolo_detection
import qwen_tensorrt as tensorrt

from ultralytics import YOLO


event_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

vlm_busy = False
vlm_busy_lock = threading.Lock()
last_vlm_trigger_time = 0.0
VLM_TRIGGER_COOLDOWN = 5.0


def draw_detections(frame, detections):
    output = frame.copy()

    for det in detections:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])
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

    color = (255, 255, 255)
    if analysis["has_fire"] or analysis["has_smoke"]:
        color = (0, 0, 255)

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


def resize_to_640(frame):
    return cv2.resize(frame, (640, 640))


def is_vlm_busy():
    with vlm_busy_lock:
        return vlm_busy


def set_vlm_busy(value: bool):
    global vlm_busy
    with vlm_busy_lock:
        vlm_busy = value


def build_vlm_prompt(detections, analysis):
    detection_lines = []
    for i, det in enumerate(detections):
        detection_lines.append(
            f"[{i}] class={det['class_name']} conf={det['conf']:.2f} "
            f"box=({int(det['x1'])}, {int(det['y1'])}, {int(det['x2'])}, {int(det['y2'])})"
        )

    detection_text = "\n".join(detection_lines) if detection_lines else "탐지 결과 없음"

    return f"""
당신은 건설 현장 안전 모니터링 도우미입니다.
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


def vlm_worker(runner):
    while not stop_event.is_set():
        try:
            item = event_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        set_vlm_busy(True)

        try:
            frame = item["frame"]
            detections = item["detections"]
            analysis = item["analysis"]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            prompt = build_vlm_prompt(detections, analysis)

            result_text = runner.infer(
                image_input=pil_image,
                user_text=prompt,
                max_new_tokens=64,
            )

            if result_text:
                print("\n[위험 분석 결과]")
                print(result_text)
                print()
            else:
                print("\n[위험 분석 결과] 유효한 문장을 생성하지 못했습니다.\n")

        except Exception as e:
            print(f"\n[VLM WORKER][오류] {e}\n")

        finally:
            set_vlm_busy(False)
            event_queue.task_done()


def main():
    model = YOLO("best.engine")
    runner = tensorrt.TensorRTQwenRunner(
        engine_dir="~/edgellm_work/engines/qwen3-vl-2b",
        multimodal_engine_dir="~/edgellm_work/visual_engines/qwen3-vl-2b",
        llm_inference_bin="~/TensorRT-Edge-LLM/build/examples/llm/llm_inference",
        plugin_path="~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so",
        work_dir="~/edgellm_work/runtime",
    )

    cap = cv2.VideoCapture("people_fire.mp4")
    if not cap.isOpened():
        print("영상 파일을 열 수 없습니다: people_fire.mp4")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_interval = 1.0 / fps

    print(f"[INFO] video fps: {fps:.2f}, frame interval: {frame_interval:.4f}s")

    worker = threading.Thread(target=vlm_worker, args=(runner,), daemon=True)
    worker.start()

    global last_vlm_trigger_time

    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        resize_frame = resize_to_640(frame)
        detections = yolo_detection.detect_positions_with_class_on_frame(model, resize_frame)
        analysis = analyze_detected_classes(detections)

        display_frame = draw_detections(resize_frame, detections)
        display_frame = draw_status(display_frame, analysis)
        cv2.imshow("frame", display_frame)

        if analysis["has_fire"] or analysis["has_smoke"]:
            now = time.time()
            if now - last_vlm_trigger_time >= VLM_TRIGGER_COOLDOWN:
                if not is_vlm_busy() and event_queue.empty():
                    event_queue.put(
                        {
                            "frame": resize_frame.copy(),
                            "detections": detections,
                            "analysis": analysis,
                            "timestamp": now,
                        }
                    )
                    last_vlm_trigger_time = now
                    print("[MAIN] TensorRT Qwen 위험 분석 이벤트 전달")

        elapsed = time.time() - loop_start
        remaining = frame_interval - elapsed
        delay_ms = max(1, int(remaining * 1000))

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == 27:
            break

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()