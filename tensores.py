from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.preprocessing import image

  # Normalización para MobileNetV2

def prediccion(model, img_array):
    # Hacer la predicción
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]  # Top 3 predicciones
    """ for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        return (f"{i+1}: {label} ({score:.2f})") """
    print(decoded_predictions[0])
    print(decoded_predictions[0][1])
    print(decoded_predictions[0][2])
    score = f"{decoded_predictions[0][2]:.2f}"
    data = {
        "TypeOfCancer": decoded_predictions[0][1],
        "confidence": score
    }
    return data
    

def modelo():
    model = load_model("modelo_perros_gatos.h5")
    return model
def imagenes(img_path):
    # Cargar una imagen de perro o gato
    #img_path = "OIP.jpg"  # Reemplaza con la imagen que quieres probar
    img = image.load_img(img_path, target_size=(224, 224))  # MobileNetV2 usa 224x224
    # Preprocesar la imagen
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Agregar dimensión batch
    img_array = preprocess_input(img_array)
    return img_array

# Decodificar las predicciones

