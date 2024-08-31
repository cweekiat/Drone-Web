from flask import Flask, request, jsonify, send_file
import cv2
import torch
from ultralytics import YOLO
import os
import logging
from werkzeug.utils import secure_filename
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Load the model (make sure to use the appropriate path)
MODEL_PATH = "models/yolov8s.pt"
DEVICE = "auto"

def get_device():
    if DEVICE == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    return DEVICE

def load_model(model_path, device):
    if not os.path.exists(model_path):
        logging.error(f"Model file {model_path} does not exist.")
        raise FileNotFoundError(f"Model file {model_path} not found.")
    model = YOLO(model_path).to(device)
    logging.info(f"Model loaded from {model_path} and sent to {device.upper()}.")
    return model

model = load_model(MODEL_PATH, get_device())

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join("/tmp", filename)
        file.save(filepath)
        
        # Process video
        result_filepath = process_video(filepath)
        
        return send_file(result_filepath, as_attachment=True)

def process_video(video_path):
    result_path = "/tmp/annotated_video.mp4"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video file {video_path}.")
        raise ValueError(f"Cannot open video file {video_path}.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    track_history = defaultdict(lambda: deque(maxlen=int(5 / (1.0 / fps))))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, verbose=False)

        for result in results:
            for track in result.boxes:
                track_id = int(track.id.item())
                bbox = track.xywh[0].cpu().numpy()
                center_x, center_y = int(bbox[0]), int(bbox[1])

                track_history[track_id].append((center_x, center_y))

        for track_id, history in track_history.items():
            for i in range(1, len(history)):
                cv2.line(frame, history[i-1], history[i], (255, 255, 255), 2)

        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()
    logging.info("Video processing completed.")
    
    return result_path

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)