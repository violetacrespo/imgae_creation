# -*- coding: utf-8 -*-
"""Utilities for image generation, YOLO analysis and result packaging."""

from pathlib import Path
import datetime
import zipfile
from PIL import Image
import numpy as np
import torch


def generar_imagen(pipe, prompt, negative_prompt, steps, cfg, seed, rescale):
    """Generate a single image with Stable Diffusion."""
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
        guidance_rescale=rescale,
    ).images[0]
    return image


def analizar_imagen(model, image_path):
    """Analyze a saved image file with YOLO and return mean confidence and object count."""

    results = model(image_path, verbose=False)
    num_objects = len(results[0].boxes)
    confidences = results[0].boxes.conf.cpu().numpy()
    mean_confidence = confidences.mean() if len(confidences) > 0 else 0
    return mean_confidence, num_objects


def analizar_imagen_memoria(model, image):
    """Analyze an in-memory image with YOLO and return mean confidence and object count."""
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    results = model(image, verbose=False)
    boxes = results[0].boxes
    num_objects = len(boxes)
    confidences = boxes.conf.cpu().numpy() if num_objects > 0 else np.array([])
    mean_confidence = float(confidences.mean()) if num_objects > 0 else 0.0
    return mean_confidence, num_objects


def ensure_dir(path):
    """Create a directory if it does not exist."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def crear_output_folder(base_dir=None):
    """Create a timestamped output folder under the provided base directory."""
    if base_dir is None:
        base_dir = Path.cwd()

    output_root = Path(base_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder = output_root / f"imagenes_{current_time}"
    output_folder.mkdir(parents=True, exist_ok=True)
    return str(output_folder), current_time


def guardar_zip_contenido(folder_path, zip_filename, output_dir=None):
    """Compress folder contents into a ZIP file under the given output directory."""
    folder_path = Path(folder_path)
    if output_dir is None:
        output_dir = folder_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / zip_filename
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files_list in os.walk(folder_path):
            for file_name in files_list:
                file_path = Path(root) / file_name
                arcname = file_path.relative_to(folder_path)
                zipf.write(file_path, arcname)

    print(f"Archivo ZIP creado: {zip_path}")
    return str(zip_path)
