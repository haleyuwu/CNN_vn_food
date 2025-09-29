# app.py (robust loader: CNN preferred, TM fallback)
import os, numpy as np, cv2
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model

# --------- CANDIDATES (ưu tiên cái đầu tồn tại) ----------
MODEL_CANDIDATES = [
    os.path.join("models", "cnn_vnfoods.h5"),                           # CNN bạn train
    r"C:\New folder\lonai\fud\converted_keras\keras_model.h5",          # fallback: Teachable Machine
]
LABELS_NPY_CANDIDATES = [
    os.path.join("models", "labels.npy"),
]
LABELS_TXT_CANDIDATES = [
    os.path.join("models", "labels.txt"),
    r"C:\New folder\lonai\fud\converted_keras\labels.txt",
]

UPLOAD_DIR  = "uploads"
ALLOWED_EXT = {"png","jpg","jpeg","bmp"}

app = Flask(__name__)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- Chọn file tồn tại ----------
def pick_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

MODEL_PATH = pick_first_existing(MODEL_CANDIDATES)
if not MODEL_PATH:
    raise RuntimeError("Không tìm thấy model. Hãy train CNN (models/cnn_vnfoods.h5) "
                       "hoặc đặt converted_keras/keras_model.h5 đúng đường dẫn.")

# ---------- Load model ----------
try:
    model = load_model(MODEL_PATH, compile=False)
except Exception as e:
    raise RuntimeError(f"Load model lỗi: {e}")

# ---------- Suy ra IMG_SIZE từ input_shape ----------
def infer_img_size(keras_model):
    shp = keras_model.input_shape
    # Có thể là tuple hoặc list (None, H, W, C) hoặc (None, C, H, W)
    if isinstance(shp, (list, tuple)):
        if isinstance(shp[0], (list, tuple)):
            shp = shp[0]
        # tìm 2 chiều liên tiếp > 20 (để loại None, C=3)
        for i in range(len(shp)-2):
            a, b = shp[i+1], shp[i+2]
            if isinstance(a, int) and isinstance(b, int) and a > 20 and b > 20:
                return int(a), int(b)
    # fallback an toàn
    return 64, 64

IMG_H, IMG_W = infer_img_size(model)
print(f"[READY] MODEL={MODEL_PATH} | INPUT_SIZE=({IMG_H},{IMG_W})")

# ---------- Load labels ----------
def load_labels():
    npy = pick_first_existing(LABELS_NPY_CANDIDATES)
    if npy:
        return list(np.load(npy, allow_pickle=True))
    txt = pick_first_existing(LABELS_TXT_CANDIDATES)
    if txt:
        out = []
        with open(txt, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split(" ", 1)   # "0 banh mi" -> ["0","banh mi"]
                out.append(parts[1] if len(parts) == 2 else parts[0])
        return out
    raise RuntimeError("Không tìm thấy labels (labels.npy hoặc labels.txt).")

try:
    classes = load_labels()
except Exception as e:
    raise RuntimeError(f"Load labels lỗi: {e}")

print(f"[READY] {len(classes)} classes:", classes)

def allowed_file(name:str)->bool:
    return "." in name and name.rsplit(".",1)[1].lower() in ALLOWED_EXT

def preprocess(file_storage):
    data = np.frombuffer(file_storage.read(), np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32")/255.0
    return np.expand_dims(img, 0)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", class_names=list(classes))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error":"No file part 'image'"}), 400
        f = request.files["image"]
        if not f.filename:
            f.filename = "frame.jpg"
        if not allowed_file(f.filename):
            return jsonify({"error":"File type not allowed"}), 400

        x = preprocess(f)
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        top3 = probs.argsort()[-3:][::-1]
        return jsonify({
            "prediction": str(classes[idx]),
            "confidence": float(probs[idx]),
            "top3": [{"label": str(classes[i]), "prob": float(probs[i])} for i in top3]
        }), 200
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

@app.errorhandler(404)
def nf(_): return jsonify({"error":"Route not found"}), 404

@app.errorhandler(405)
def mm(_): return jsonify({"error":"Method not allowed"}), 405

if __name__ == "__main__":
    # Mở bằng http://127.0.0.1:5000 để bật được camera từ trình duyệt
    app.run(host="0.0.0.0", port=5000, debug=True)
