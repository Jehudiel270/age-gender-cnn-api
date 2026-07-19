import io

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

IMG_HEIGHT = 128
IMG_WIDTH = 128
AGE_NORM_FACTOR = 100.0
MODEL_PATH = "model/AgeGenderCNN_v5.tflite"
FACE_MARGIN_RATIO = 0.35

app = FastAPI(title="Age & Gender Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://age-gender-ai.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def detect_and_crop_face(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return image_rgb

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    margin_x = int(w * FACE_MARGIN_RATIO)
    margin_y = int(h * FACE_MARGIN_RATIO)

    img_h, img_w = image_rgb.shape[:2]
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)

    return image_rgb[y1:y2, x1:x2]


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")

    image_np = np.array(image)
    face_np = detect_and_crop_face(image_np)

    face_image = Image.fromarray(face_np)
    face_image = face_image.resize((IMG_WIDTH, IMG_HEIGHT))

    array = np.array(face_image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)
    return array


def run_tflite_inference(input_array: np.ndarray):
    """Renvoie (age_raw, gender_raw). L'ordre observé empiriquement
    pour ce modèle est [gender, age] (sortie 0 = gender, sortie 1 = age) —
    déduit du test manuel des noms de sorties TFLite (StatefulPartitionedCall_1:1
    avant :0). Si les résultats semblent incohérents en usage réel,
    inverse simplement gender_raw et age_raw ci-dessous."""
    interpreter.set_tensor(input_details[0]["index"], input_array)
    interpreter.invoke()

    gender_raw = float(interpreter.get_tensor(output_details[0]["index"])[0][0])
    age_raw = float(interpreter.get_tensor(output_details[1]["index"])[0][0])

    return age_raw, gender_raw


@app.get("/")
def root():
    return {"status": "ok", "message": "Age & Gender Prediction API (TFLite)"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier envoyé n'est pas une image.")

    try:
        image_bytes = await file.read()
        input_array = preprocess_image(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de lire l'image envoyée.")

    try:
        age_raw, gender_raw = run_tflite_inference(input_array)
    except Exception:
        raise HTTPException(status_code=500, detail="Erreur lors de la prédiction.")

    age_value = age_raw * AGE_NORM_FACTOR
    gender_label = "Femme" if gender_raw > 0.5 else "Homme"
    gender_confidence = abs(gender_raw - 0.5) * 2

    return {
        "age": round(age_value, 1),
        "gender": gender_label,
        "gender_confidence": round(gender_confidence, 3),
    }