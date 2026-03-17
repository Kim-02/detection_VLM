from ultralytics import YOLO

#img
# def detect_positions_with_class(engine_path: str, image_path: str):
#     model = YOLO(engine_path)

#     results = model.predict(
#         source=image_path,
#         imgsz=640,
#         conf=0.25,
#         iou=0.45,
#         verbose=False,
#         device=0
#     )

#     result = results[0]
#     boxes = result.boxes.xyxy.cpu().numpy()
#     classes = result.boxes.cls.cpu().numpy()
#     confs = result.boxes.conf.cpu().numpy()

#     detections = []
#     for box, cls_id, conf in zip(boxes, classes, confs):
#         x1, y1, x2, y2 = box.tolist()
#         detections.append({
#             "class_id": int(cls_id),
#             "conf": float(conf),
#             "x1": float(x1),
#             "y1": float(y1),
#             "x2": float(x2),
#             "y2": float(y2),
#         })

#     return detections

def detect_positions_with_class_on_frame(model, frame):
    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        verbose=False,
        device=0
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    names = result.names

    detections = []
    for box, cls_id, conf in zip(boxes, classes, confs):
        x1, y1, x2, y2 = box.tolist()
        detections.append({
            "class_id": int(cls_id),
            "class_name": names[int(cls_id)],
            "conf": float(conf),
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
        })

    return detections