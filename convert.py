from ultralytics import YOLO

# Charger le modèle YOLO
model = YOLO("best.pt")

# Convertir le modèle en TFLite
model.export(format="tflite")
