import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

# Cargar el modelo preentrenado con pesos de ImageNet
model = MobileNetV2(weights="imagenet")

# Guardar el modelo en un archivo .h5
model.save("modelo_perros_gatos.h5")
