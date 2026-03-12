"""
Video Analyzer Lite
Real-time gender & age prediction using OpenCV DNN.
Usage: python analyzer.py
       python analyzer.py --source video.mp4
       python analyzer.py --source video.mp4 --output result.mp4
"""

import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# ── Model file paths ──────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "models"

FACE_PROTO  = MODELS_DIR / "deploy.prototxt"
FACE_MODEL  = MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
AGE_PROTO   = MODELS_DIR / "age_deploy.prototxt"
AGE_MODEL   = MODELS_DIR / "age_net.caffemodel"
GENDER_PROTO = MODELS_DIR / "gender_deploy.prototxt"
GENDER_MODEL = MODELS_DIR / "gender_net.caffemodel"

# ── Age / gender labels ───────────────────────────────────────────────────────
AGE_BUCKETS  = ["0-2","4-6","8-12","15-20","25-32","38-43","48-53","60+"]
GENDER_LIST  = ["Male", "Female"]
MODEL_MEAN   = (78.4263377603, 87.7689143744, 114.895847746)

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
MALE_COLOR   = (255, 160, 50)
FEMALE_COLOR = (200, 80, 200)


# ─────────────────────────────────────────────────────────────────────────────
# Model download
# ─────────────────────────────────────────────────────────────────────────────

DOWNLOAD_URLS = {
    "deploy.prototxt":
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel":
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "age_deploy.prototxt":
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
    "age_net.caffemodel":
        "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/age_net.caffemodel",
    "gender_deploy.prototxt":
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
    "gender_net.caffemodel":
        "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel",
}


def download_models():
    MODELS_DIR.mkdir(exist_ok=True)
    print("\n Downloading models...\n")
    for filename, url in DOWNLOAD_URLS.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"   {filename} (already exists)")
            continue
        print(f"   {filename} ...", end="", flush=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VideoAnalyzerLite"})
            with urllib.request.urlopen(req, timeout=60) as r, dest.open("wb") as f:
                total = int(r.headers.get("Content-Length", 0))
                done  = 0
                while chunk := r.read(65536):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  ↓ {filename} ... {done/total*100:.0f}%  ", end="", flush=True)
            print(f"\r   {filename} ({dest.stat().st_size//1024:,} KB)")
        except Exception as e:
            print(f"\r   {filename} – {e}")
    print("\n Done! Run: python analyzer.py\n")


def models_ready():
    return all(p.exists() for p in [FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL])


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────────────────────────────────────

def load_networks():
    face_net   = cv2.dnn.readNet(str(FACE_MODEL),   str(FACE_PROTO))
    age_net    = cv2.dnn.readNet(str(AGE_MODEL),    str(AGE_PROTO))
    gender_net = cv2.dnn.readNet(str(GENDER_MODEL), str(GENDER_PROTO))
    return face_net, age_net, gender_net


def detect_faces(frame, net, confidence_threshold=0.7):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                  (300, 300), (104, 177, 123))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < confidence_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        if (x2-x1) > 20 and (y2-y1) > 20:
            faces.append((x1, y1, x2, y2, conf))
    return faces


def predict_age_gender(face_crop, age_net, gender_net):
    blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), MODEL_MEAN)

    gender_net.setInput(blob)
    gender = GENDER_LIST[np.argmax(gender_net.forward())]

    age_net.setInput(blob)
    age = AGE_BUCKETS[np.argmax(age_net.forward())]

    return gender, age


def draw_label(frame, x1, y1, x2, y2, gender, age):
    color = MALE_COLOR if gender == "Male" else FEMALE_COLOR
    label = f"{gender} | Age {age}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    bg_y1 = y1 - th - 10
    bg_y2 = y1
    if bg_y1 < 0:
        bg_y1, bg_y2 = y2, y2 + th + 10

    cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 8, bg_y2), color, cv2.FILLED)
    cv2.putText(frame, label, (x1 + 4, bg_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run(source, output_path=None):
    face_net, age_net, gender_net = load_networks()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Cannot open: {source}")
        return

    writer = None
    if output_path:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f" Saving to: {output_path}")

    print(" Running — press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame, face_net)

        for (x1, y1, x2, y2, _) in faces:
            # Add a little padding around the face crop
            pad = 20
            fx1 = max(0, x1 - pad)
            fy1 = max(0, y1 - pad)
            fx2 = min(frame.shape[1]-1, x2 + pad)
            fy2 = min(frame.shape[0]-1, y2 + pad)
            crop = frame[fy1:fy2, fx1:fx2]

            if crop.size == 0:
                continue

            gender, age = predict_age_gender(crop, age_net, gender_net)
            draw_label(frame, x1, y1, x2, y2, gender, age)

        if writer:
            writer.write(frame)

        cv2.imshow("Video Analyzer Lite — press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(" Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Analyzer Lite")
    parser.add_argument("--source",  default="0",  help="Webcam index or video file path")
    parser.add_argument("--output",  default=None, help="Save output to this .mp4 file")
    parser.add_argument("--download-models", action="store_true", help="Download model files")
    args = parser.parse_args()

    if args.download_models:
        download_models()
    else:
        if not models_ready():
            print("  Models not found. Downloading now...\n")
            download_models()

        source = int(args.source) if args.source.isdigit() else args.source
        run(source, args.output)
