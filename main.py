import io

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow import keras

# ============================================================
# Configuration — doit correspondre EXACTEMENT à l'entraînement
# ============================================================
IMG_HEIGHT = 128
IMG_WIDTH = 128
AGE_NORM_FACTOR = 100.0
MODEL_PATH = "model/AgeGenderCNN_v5.keras"

# Marge ajoutée autour du visage détecté, en % de la taille de la boîte
# détectée (évite un recadrage trop serré qui couperait le menton/front).
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

# Modèle de détection de visage OpenCV (Haar Cascade), inclus avec le paquet.
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Chargement du modèle Keras une seule fois, au démarrage du serveur.
model = keras.models.load_model(MODEL_PATH)


def detect_and_crop_face(image_rgb: np.ndarray) -> np.ndarray:
    """
    Détecte le plus grand visage dans l'image et le recadre avec une marge,
    pour reproduire le cadrage 'aligned & cropped' utilisé par UTKFace.
    Si aucun visage n'est détecté, renvoie l'image entière inchangée
    (fallback : mieux vaut une prédiction dégradée qu'une erreur).
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        return image_rgb

    # On garde le plus grand visage détecté (le sujet principal probable)
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
    """Détecte et recadre le visage, puis reproduit le preprocessing
    utilisé à l'entraînement : RGB, resize 128x128, normalisation /255,
    ajout de la dimension batch."""
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")  # force 3 canaux, même si upload PNG/grayscale

    image_np = np.array(image)
    face_np = detect_and_crop_face(image_np)

    face_image = Image.fromarray(face_np)
    face_image = face_image.resize((IMG_WIDTH, IMG_HEIGHT))

    array = np.array(face_image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)  # (128,128,3) -> (1,128,128,3)
    return array


@app.get("/")
def root():
    return {"status": "ok", "message": "Age & Gender Prediction API"}


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
        age_pred, gender_pred = model.predict(input_array, verbose=0)
    except Exception:
        raise HTTPException(status_code=500, detail="Erreur lors de la prédiction.")

    # Dénormalisation de l'âge (inverse de /100 fait à l'entraînement)
    age_value = float(age_pred[0][0]) * AGE_NORM_FACTOR

    # Sortie sigmoid : 0 = homme, 1 = femme (encodage de ton dataset)
    gender_proba = float(gender_pred[0][0])
    gender_label = "Femme" if gender_proba > 0.5 else "Homme"

    # Confiance du genre : distance au seuil 0.5, ramenée entre 0 et 1
    gender_confidence = abs(gender_proba - 0.5) * 2

    return {
        "age": round(age_value, 1),
        "gender": gender_label,
        "gender_confidence": round(gender_confidence, 3),
    }