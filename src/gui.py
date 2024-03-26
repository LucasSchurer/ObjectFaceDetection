from flask import Flask, render_template, request, jsonify
import base64
from model import Model
from threading import Lock
import cv2
import numpy as np
from flask import Blueprint

app = Flask(__name__)
static_bp = Blueprint('static', __name__, static_folder='static', static_url_path='/static')
app.register_blueprint(static_bp)
lock = Lock()

model = Model()
model.load('../models/yolov8n.pt')

@app.route('/')
def index() :
    return render_template('index.html')

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    with lock :
        frame_data = request.json.get('frame')
        content = frame_data.split(';')

        plot_classification = request.json.get('plot_classification', False)
        plot_object_confidence = request.json.get('plot_object_confidence', False)
        
        detect_faces = request.json.get('detect_faces', False)
        plot_face_distance = request.json.get('plot_face_distance', False)
        stop_first_match = request.json.get('stop_first_match', False)
        show_only_best_match = request.json.get('show_only_best_match', False)

        if len(content) > 1 :
            content = content[1]
            encoded_data = content.split(',')[1]
            decoded_image = base64.b64decode(encoded_data)

            nparr = np.frombuffer(decoded_image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.flip(img, 1)

            if detect_faces :
                result = model.detect_all(img, plot_classification=plot_classification, plot_object_confidence=plot_object_confidence, 
                                          plot_face_distance=plot_face_distance, stop_first_match=stop_first_match, show_only_best_match=show_only_best_match)
            else :
                result = model.detect_objects(img).plot()

            _, encoded_image  = cv2.imencode('.jpg', result)

            encoded_image = base64.b64encode(encoded_image).decode('utf-8')

            return jsonify({
                'processed_frame': 'data:image/png;base64,' + encoded_image, 
                'frame': request.json.get('frame'),
            })

        return jsonify({
            'processed_frame': 'https://pm1.aminoapps.com/6475/e3b27c0e80d5323b18550ec43a2b1e8e8731ab4f_hq.jpg',
            'frame': request.json.get('frame'),
        })

@app.route('/save_face', methods=['POST'])
def save_face() :
    face_name = request.json.get('name')
    face_data = request.json.get('face')
    content = face_data.split(';')
    sucess = False    

    if len(content) > 1 and face_name is not None :
        content = content[1]
        encoded_data = content.split(',')[1]
        decoded_image = base64.b64decode(encoded_data)

        nparr = np.frombuffer(decoded_image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.flip(img, 1)

        model.detect_save_face_embedding(face_name, img)

        sucess = True

    return jsonify({
        'success': sucess
    })

if __name__ == '__main__' :
    app.run(debug=True, port='5000')