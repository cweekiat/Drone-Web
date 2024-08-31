from flask import Flask, request, jsonify
import cv2
import torch
from ultralytics import YOLO
import os
import logging
from collections import defaultdict, deque
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Load the model
model_path = "models/yolov8s.pt"
device = "cpu"  # Adjust this if you have GPU support
if not os.path.exists(model_path):
    logging.error(f"Model file {model_path} does not exist.")
    raise FileNotFoundError(f"Model file {model_path} not found.")
model = YOLO(model_path).to(device)

def get_device(device_choice):
    if device_choice == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    return device_choice

@app.route('/process', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    video_path = os.path.join("/tmp", video_file.filename)
    video_file.save(video_path)

    if not os.path.exists(video_path):
        return jsonify({"error": f"Video file {video_path} does not exist."}), 404

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": f"Failed to open video file {video_path}."}), 400

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        track_history = defaultdict(lambda: deque(maxlen=int(5 / (1.0 / fps))))

        with tqdm(total=frame_count, desc="Processing video") as pbar:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                start_time = time.time()

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

                end_time = time.time()
                fps = 1.0 / (end_time - start_time)
                cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                pbar.update(1)

        cap.release()
        cv2.destroyAllWindows()

        # Optionally save the processed video or send results back
        # For now, we'll just return a success message
        return jsonify({"message": "Video processed successfully"})
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)