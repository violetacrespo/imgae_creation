# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from utils import generar_imagen

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from utils import generar_imagen

def procesar_resultados_evolutivos(objeto, pipe):
    base_path = f"resultados/{objeto}"
    summary_path = f"{base_path}/runs_summary.csv"
    plot_dir = f"{base_path}/plots1"
    os.makedirs(plot_dir, exist_ok=True)
    
    df = pd.read_csv(summary_path)
    hv_mean, hv_std = df["hypervolume_feas"].mean(), df["hypervolume_feas"].std()
    
    # Normalización para selección de mejor run
    df["hv_norm"] = (df["hypervolume_feas"] - df["hypervolume_feas"].min()) / (df["hypervolume_feas"].max() - df["hypervolume_feas"].min())
    best_run_row = df.loc[df["hv_norm"].idxmax()]
    run_id_str = str(int(best_run_row["run_id"])).zfill(2)

    # Cargar Pareto
    pareto_path = f"{base_path}/paretos/pareto_{objeto}_sbx_polynomial_run{run_id_str}.csv"
    pf = pd.read_csv(pareto_path)

    # --- LÓGICA KNEE POINT (Distancia al punto ideal) ---
    # Normalizamos f2 y fitness_yolo para que la distancia sea justa (escala 0-1)
    f2_norm = (pf["f2"] - pf["f2"].min()) / (pf["f2"].max() - pf["f2"].min())
    yolo_norm = (pf["fitness_yolo"] - pf["fitness_yolo"].min()) / (pf["fitness_yolo"].max() - pf["fitness_yolo"].min())
    
    # Punto ideal en nuestro caso: f2 mínimo (0) y yolo máximo (1)
    distancias = np.sqrt((f2_norm - 0)**2 + (yolo_norm - 1)**2)
    idx_knee = distancias.idxmin()
    idx_best_yolo = pf["fitness_yolo"].idxmax()
    
    # --- PLOT DE LA FRONTERA CON AMBOS PUNTOS ---
    pf_sorted = pf.sort_values("f2")
    plt.figure(figsize=(8, 5))
    plt.plot(pf_sorted["f2"], pf_sorted["fitness_yolo"], linestyle='--', alpha=0.5, color='gray')
    plt.scatter(pf["f2"], pf["fitness_yolo"], c='blue', label='Otras soluciones', alpha=0.3)
    
    # Resaltar puntos seleccionados
    plt.scatter(pf.loc[idx_best_yolo, "f2"], pf.loc[idx_best_yolo, "fitness_yolo"], 
                color='red', s=100, label='Mejor YOLO', edgecolors='black')
    plt.scatter(pf.loc[idx_knee, "f2"], pf.loc[idx_knee, "fitness_yolo"], 
                color='green', s=100, label='Knee Point (Equilibrio)', edgecolors='black')
    
    plt.xlabel("f2 (Minimizar)")
    plt.ylabel("Fitness YOLO (Maximizar)")
    plt.title(f"Frontera de Pareto - {objeto} (Run {run_id_str})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_dir}/pareto_frontera_seleccion.png")
    plt.close()

    # --- GENERACIÓN DE IMÁGENES ---
    configuraciones = [
        ("mejor_yolo", pf.loc[idx_best_yolo]),
        ("knee_point", pf.loc[idx_knee])
    ]

    for nombre, ind in configuraciones:
        print(f"Generando imagen {nombre} para {objeto}...")
        img = generar_imagen(
            pipe=pipe,
            prompt=f"{objeto}, photograph, digital, color, blended visuals",
            negative_prompt="illustration, painting, drawing, art",
            steps=int(round(ind["iterations"])),
            cfg=float(ind["cfg"]),
            seed=int(round(ind["sd_seed"])),
            rescale=float(ind["guidance_rescale"])
        )
        img.save(f"{plot_dir}/img_{nombre}.png")

    # Guardar métricas
    pd.DataFrame({"objeto": [objeto], "hv_mean": [hv_mean], "hv_std": [hv_std]}).to_csv(f"{plot_dir}/metricas_hv.csv", index=False)
    
    print(f"✓ Finalizado {objeto}: 2 imágenes generadas.")
    return True
