from flask import Flask, render_template, request, jsonify
import base64
from model import Model
from threading import Lock
import cv2
import numpy as np
from flask import Blueprint
from predict import detect_image
from dotenv import load_dotenv

app = Flask(__name__)
static_bp = Blueprint(
    "static", __name__, static_folder="static", static_url_path="/static"
)
app.register_blueprint(static_bp)
lock = Lock()

load_dotenv()
model = Model()
model.load("../models/yolov8n.pt")


def get_request_image(img_str: str) -> np.ndarray:
    if img_str is None:
        return None

    content = img_str.split(";")

    if len(content) > 1:
        content = content[1]
        encoded_data = content.split(",")[1]
        decoded_image = base64.b64decode(encoded_data)

        nparr = np.frombuffer(decoded_image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        return img
    else:
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    with lock:
        img_str = request.json.get("frame")

        img = get_request_image(img_str)

        if img is not None:
            detect_objects = request.json.get("detect_objects", False)
            object_detection_threshold = float(
                request.json.get("object_detection_threshold", 0.7)
            )

            object_draw_classification = request.json.get(
                "object_draw_classification", True
            )
            object_draw_confidence = request.json.get("object_draw_confidence", True)

            detect_faces = request.json.get("detect_faces", False)
            face_detection_threshold = float(
                request.json.get("face_detection_threshold", 0.7)
            )
            face_detection_draw_confidence = request.json.get(
                "face_detection_draw_confidence", True
            )

            recognize_faces = request.json.get("recognize_faces", False)
            face_recognition_match_type = request.json.get("recognize_faces", "all")
            face_recognition_threshold = float(request.json.get("recognize_faces", 0.7))

            result = detect_image(
                model,
                img,
                detect_objects=detect_objects,
                object_detection_threshold=object_detection_threshold,
                object_detection_draw_classification=object_draw_classification,
                object_detection_draw_confidence=object_draw_confidence,
                detect_faces=detect_faces,
                face_detection_threshold=face_detection_threshold,
                face_detection_draw_confidence=face_detection_draw_confidence,
                recognize_faces=recognize_faces,
                face_recognition_match_type=face_recognition_match_type,
                face_recognition_threshold=face_recognition_threshold,
            )

            _, encoded_image = cv2.imencode(".jpg", result)

            encoded_image = base64.b64encode(encoded_image).decode("utf-8")

            return jsonify(
                {
                    "processed_frame": "data:image/png;base64," + encoded_image,
                    "frame": request.json.get("frame"),
                }
            )
        else:
            return jsonify(
                {
                    "processed_frame": "",
                    "frame": request.json.get("frame"),
                }
            )


@app.route("/save_face", methods=["POST"])
def save_face():
    success = False
    face_name = request.json.get("name", None)
    img_str = request.json.get("face", None)

    img = get_request_image(img_str)

    if img is not None and face_name is not None:
        success = model.recognize_save_face(face_name, img)

    return jsonify({"success": success})


if __name__ == "__main__":
    app.run(debug=True, port="5000")
