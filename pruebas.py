from fastapi import FastAPI, File, UploadFile
import shutil
import os
import tensores as ts

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
modelo = ts.modelo()
@app.post("/upload/")
async def upload_images(file1: UploadFile = File(...)):
    file1_path = os.path.join(UPLOAD_FOLDER, file1.filename)
    

    # Guardar las imágenes en el servidor
    with open(file1_path, "wb") as buffer:
        shutil.copyfileobj(file1.file, buffer)
  

    return {"message": "Imágenes subidas exitosamente", "file1": file1.filename}

@app.get("/model/{name1}")
def use_model(name1: str):
    imagen = ts.imagenes("uploads/" + name1)
    texto = ts.prediccion(modelo, imagen)
    print(texto)
    return {"tipoCancer":texto["TypeOfCancer"],
            "confianza":texto["confidence"]}


