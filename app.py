import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Charger le modèle YOLO
model = YOLO("best.pt")  # Remplacez par le chemin correct vers votre modèle


# Fonction pour effectuer l'inférence
def run_inference(image: np.ndarray):
    st.write("Running inference...")  # Message de débogage

    # Vérifiez que l'image est bien en uint8 (format supporté par OpenCV)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Dimensions originales de l'image
    original_height, original_width = image.shape[:2]

    # Redimensionner l'image pour le modèle
    image_resized = cv2.resize(
        image, (640, 640)
    )  # Adapter la taille à celle attendue par YOLOv8
    results = model(image_resized)

    # Extraire les boîtes englobantes, labels, et scores
    boxes = results[
        0
    ].boxes.xyxy.numpy()  # Coordonnées des boîtes englobantes (x1, y1, x2, y2)
    labels = results[0].boxes.cls.numpy()  # Indices des classes détectées
    scores = results[0].boxes.conf.numpy()  # Scores de confiance

    # Ajuster les coordonnées des boîtes à l'image originale
    scale_x = original_width / 640
    scale_y = original_height / 640
    boxes[:, [0, 2]] *= scale_x  # Ajuster x1 et x2
    boxes[:, [1, 3]] *= scale_y  # Ajuster y1 et y2

    return boxes, labels, scores


# Titre de l'application
st.title("WECCET INTERFACE")

# Option pour télécharger une image
uploaded_image = st.file_uploader("Téléchargez une image", type=["jpg", "png", "jpeg"])

# Option pour capturer une image via webcam
# webcam_image = st.camera_input("Ou capturez une image via votre webcam")

# Initialiser l'image à None
image = None

# Si aucune image n'est fournie par l'utilisateur, charger l'image par défaut
if uploaded_image is not None:
    st.write("Image uploaded.")  # Message de débogage
    image = Image.open(uploaded_image).convert("RGB")


if image is not None:
    st.image(image, caption="Image originale", use_column_width=True)
    image_np = np.array(image)

    # Effectuer l'inférence
    boxes, labels, scores = run_inference(image_np)

    # Dessiner les boîtes englobantes et étiquettes sur l'image
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image_np,
            f"{model.names[int(label)]}: {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    st.image(image_np, caption="Image avec détection", use_column_width=True)
else:
    st.write(
        "Aucune image n'a été chargée ou capturée."
    )  # Message si aucune image n'est disponible
