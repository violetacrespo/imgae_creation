# -*- coding: utf-8 -*-
import os
import datetime
import numpy as np
import torch
from pymoo.core.problem import ElementwiseProblem
from diffusers import StableDiffusionPipeline
from ultralytics import YOLO
from utils import generar_imagen, analizar_imagen, guardar_zip_contenido, analizar_imagen_memoria
from operadores import get_crossover, get_mutation

# Cargar modelos (una sola vez)
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("yolov8n.pt")

class StableDiffusionProblem(ElementwiseProblem):
    def __init__(self, save_images=False, **kwargs):
        
        self.save_images = save_images
        
        # Rangos de los parámetros
        ITERATIONS_RANGE = (1, 100)  # Número de iteraciones
        CFG_RANGE = (1, 20)  # Escala de orientación
        SEED_RANGE = (0, 10000)  # Semilla
        GUIDANCE_RESCALE_RANGE = (0, 1)  # Reducción de la orientación

        # Límites inferiores y superiores para cada variable
        xl = np.array([ITERATIONS_RANGE[0], CFG_RANGE[0], SEED_RANGE[0], GUIDANCE_RESCALE_RANGE[0]])
        xu = np.array([ITERATIONS_RANGE[1], CFG_RANGE[1], SEED_RANGE[1], GUIDANCE_RESCALE_RANGE[1]])

        # Llamar al constructor de la clase base
        super().__init__(n_var=4,  # Número de variables de decisión
                         n_obj=2,  # Número de funciones objetivo
                         n_constr=1,  # Número de restricciones 1, ya que tengo que limitar la rate de yolo 
                         xl=xl,  # Límites inferiores
                         xu=xu,
                         **kwargs # aquí entrara el elementwise se supnone
                        )  # Límites superiores
        
        # Carpeta donde está este archivo problema.py
        project_root = os.path.dirname(os.path.abspath(__file__))

        # Aquí creará: violetatfg/TFG/imagenes_generadas
        self.base_path = None
        if self.save_images:
            self.base_path = os.path.join(project_root, "TFG", "imagenes_generadas")
            os.makedirs(self.base_path, exist_ok=True)

        # Prompts fijos
        self.prompt = "golden retriever dog, photograph, digital, color, blended visuals"
        self.negative_prompt = "illustration, painting, drawing, art"

    def _evaluate(self, x, out, *args, **kwargs):
        # Crear subcarpeta única para cada evaluación
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')#añadimos microsegundos para que no colisione el algoritmo cuando se paralelice la evaluación de las imágenes con Yolo. 

        carpeta_salida = None
        if self.save_images:
            carpeta_salida = os.path.join(self.base_path, f"imagenes_{timestamp}")
            os.makedirs(carpeta_salida, exist_ok=True)

        
        iterations = int(x[0])
        cfg_scale = float(x[1])
        seed = int(x[2])
        guidance_rescale = float(x[3])

        image = generar_imagen(
            pipe=pipe,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            steps=iterations,
            cfg=cfg_scale,
            seed=seed,
            rescale=guidance_rescale
        )

        if self.save_images:
            image_path = os.path.join(carpeta_salida, f"imagen.png")
            image.save(image_path)
            fitness_yolo, num_objects = analizar_imagen(model, image_path)
        else:
            fitness_yolo, num_objects = analizar_imagen_memoria(model, image)


        f1=-fitness_yolo
        f2=iterations
        g1 = 0.1 - fitness_yolo

                    
        if self.save_images:
            zip_filename = f"{timestamp}.zip"
            guardar_zip_contenido(carpeta_salida, zip_filename)

        out["F"] = np.array([f1,f2])
        out["G"] = np.array([g1])


