import os
import ssl
import time
import urllib.request
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = "yolo11n.pt"                        # Change to "./models/best.pt" if using custom model
CAMERA_SOURCE = "https://10.76.24.67:8080/video" # Update IP if your phone's IP changes
OUTPUT_DIR = "./fetched_images"
TARGET_CLASSES = None                            # None = detect all; or e.g. ["person", "car"]
CONFIDENCE_THRESHOLD = 0.5
SAVE_CROPPED_ONLY = False
SAVE_COOLDOWN = 2.0
SHOW_PREVIEW = True

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)

# Bypass self-signed SSL certificate verification
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

print(f"Connecting to {CAMERA_SOURCE}...")
try:
    stream = urllib.request.urlopen(CAMERA_SOURCE, context=ssl_ctx, timeout=10)
except Exception as e:
    raise RuntimeError(f"Could not connect to {CAMERA_SOURCE}: {e}")

print("Connected! Press 'q' in the video window to quit.")
byte_buffer = b""
last_saved_time = 0

while True:
    chunk = stream.read(4096)
    if not chunk:
        print("Stream ended.")
        break
    byte_buffer += chunk

    # Extract complete JPEG frames (starts with 0xff 0xd8, ends with 0xff 0xd9)
    start = byte_buffer.find(b"\xff\xd8")
    end = byte_buffer.find(b"\xff\xd9")

    if start != -1 and end != -1 and end > start:
        jpg_bytes = byte_buffer[start : end + 2]
        byte_buffer = byte_buffer[end + 2 :]

        frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        current_time = time.time()

        for result in results:
            for idx, box in enumerate(result.boxes):
                class_name = model.names[int(box.cls)]

                if TARGET_CLASSES is None or class_name in TARGET_CLASSES:
                    if (current_time - last_saved_time) >= SAVE_COOLDOWN:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        if SAVE_CROPPED_ONLY:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            crop = frame[y1:y2, x1:x2]
                            filename = f"{class_name}_{timestamp}_{idx}.jpg"
                            cv2.imwrite(os.path.join(OUTPUT_DIR, filename), crop)
                        else:
                            filename = f"{class_name}_{timestamp}.jpg"
                            cv2.imwrite(os.path.join(OUTPUT_DIR, filename), frame)

                        print(f"[SAVED] {filename}")
                        last_saved_time = current_time
                        if not SAVE_CROPPED_ONLY:
                            break

            if SHOW_PREVIEW:
                cv2.imshow("YOLO Camera Fetcher", result.plot())

        if SHOW_PREVIEW and (cv2.waitKey(1) & 0xFF == ord("q")):
            break

cv2.destroyAllWindows()
