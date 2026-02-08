# -*- coding: utf-8 -*-
# utils.py
import os
import datetime
import torch
import zipfile
from PIL import Image
import numpy as np

def generar_imagen(pipe, prompt, negative_prompt, steps, cfg, seed, rescale):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
        guidance_rescale=rescale
    ).images[0]

    return image


def analizar_imagen(model, image_path):
    results = model(image_path, verbose=False)
    num_objects = len(results[0].boxes)
    confidences = results[0].boxes.conf.cpu().numpy()
    mean_confidence = confidences.mean() if len(confidences) > 0 else 0
    return mean_confidence, num_objects

# +

def analizar_imagen_memoria(model, image):
    """
    Analiza una imagen en memoria (PIL.Image o numpy array) con YOLO.
    Devuelve los mismos valores que analizar_imagen.
    """
    # PIL → numpy si hace falta
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    results = model(image)

    boxes = results[0].boxes
    num_objects = len(boxes)

    if num_objects == 0:
        mean_confidence = 0.0
    else:
        confidences = boxes.conf.cpu().numpy()
        mean_confidence = float(confidences.mean())

    return mean_confidence, num_objects



# -

def crear_output_folder():
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder = os.path.join("/content", f"imagenes_{current_time}")
    os.makedirs(output_folder, exist_ok=True)
    return output_folder, current_time


def guardar_zip_contenido(folder_path, zip_filename):
    # Carpeta base en Google Drive donde guardar el zip
    base_path = "/content/drive/MyDrive/geneticos-tfg/TFG/zips_prueba2"
    
    # Asegurarse de que la carpeta existe
    os.makedirs(base_path, exist_ok=True)
    
    # Ruta completa donde se guardará el zip
    zip_path = os.path.join(base_path, zip_filename)
    
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for root, _, files_list in os.walk(folder_path):
            for file in files_list:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
    
    print(f"Archivo ZIP creado: {zip_path}")
    return zip_path
