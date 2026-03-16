import requests

url = "http://127.0.0.1:8000/infer"

detections = """[0] class=2 conf=0.799316 box=(540.75, 515.625, 998.25, 984.375)
[1] class=2 conf=0.721191 box=(1037.25, 173.25, 1416.75, 920.25)
[2] class=0 conf=0.704102 box=(1254.75, 189.375, 1368.75, 275.625)
[3] class=1 conf=0.672363 box=(1150.5, 297.375, 1360.5, 575.625)
[4] class=0 conf=0.642578 box=(1170.38, 55.6875, 1268.62, 142.312)
[5] class=0 conf=0.563965 box=(603.938, 525, 767.062, 652.5)
[6] class=2 conf=0.433838 box=(1026, 33.5625, 1293, 431.438)
[7] class=1 conf=0.416992 box=(1071.38, 106.688, 1265.62, 371.812)"""

class_map_json = """{
  "0": "helmet",
  "1": "vest",
  "2": "person",
  "3": "danger_vehicle"
}"""

with open("test.jpg", "rb") as f:
    files = {
        "file": ("test.jpg", f, "image/jpeg")
    }
    data = {
        "prompt": "장면에 대한 분석을 하되, YOLO에서 찾은 객체를 기준으로 설명해줘.",
        "detections": detections,
        "class_map_json": class_map_json
    }

    resp = requests.post(url, files=files, data=data, timeout=300)
    print(resp.status_code)
    print(resp.json())