"""Genera las gráficas del README a partir de los CSV de gradbench y gradbench_cuda.

    python bench/graficas.py --cpu resultados/cpu_ryzen7700x.csv --gpu resultados/gpu_t4.csv

Solo usa la biblioteca estándar y matplotlib.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXTO = "#0b0b0b"
TEXTO_SUAVE = "#52514e"
REJILLA = "#e4e3df"
FONDO = "#fcfcfb"


def estilo():
    plt.rcParams.update(
        {
            "figure.facecolor": FONDO,
            "axes.facecolor": FONDO,
            "axes.edgecolor": REJILLA,
            "axes.labelcolor": TEXTO_SUAVE,
            "axes.titlecolor": TEXTO,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": REJILLA,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXTO_SUAVE,
            "ytick.color": TEXTO_SUAVE,
            "legend.frameon": False,
            "font.size": 10,
            "lines.linewidth": 2,
        }
    )


def leer(ruta):
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def guardar(fig, salida, nombre):
    fig.tight_layout()
    ruta = salida / nombre
    fig.savefig(ruta, dpi=160)
    plt.close(fig)
    print(f"  {ruta}")


def escalado(filas, salida, cpu):
    datos = [f for f in filas if f["experimento"] == "escalado" and f["metodo"] == "omp_nivel1"]
    if not datos:
        return
    hilos = [int(f["hilos_ext"]) for f in datos]
    speedup = [float(f["speedup"]) for f in datos]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hilos, hilos, color=TEXTO_SUAVE, linestyle="--", linewidth=1, label="Ideal")
    ax.plot(hilos, speedup, color=SERIES[0], marker="o", markersize=7, label="OpenMP (un nivel)")
    for h, s in zip(hilos, speedup):
        ax.annotate(f"{s:.1f}x", (h, s), textcoords="offset points", xytext=(0, 8),
                    ha="center", color=TEXTO, fontsize=9)
    ax.set_xticks(hilos)
    ax.set_xlabel("Hilos")
    ax.set_ylabel("Speed-up frente al secuencial")
    ax.set_title(f"Escalabilidad con el número de hilos ({cpu})")
    ax.legend(loc="upper left")
    guardar(fig, salida, "cpu_escalado.png")


def anidado(filas, salida, cpu):
    datos = [f for f in filas if f["experimento"] == "anidado" and f["metodo"] != "secuencial"]
    if not datos:
        return
    etiquetas = [f'{f["hilos_ext"]}x{f["hilos_int"]}' for f in datos]
    speedup = [float(f["speedup"]) for f in datos]
    colores = [SERIES[1] if f["metodo"].endswith("sobresuscrito") else SERIES[0] for f in datos]
    fig, ax = plt.subplots(figsize=(7, 4))
    barras = ax.bar(etiquetas, speedup, color=colores, width=0.6, edgecolor=FONDO, linewidth=2)
    ax.bar_label(barras, fmt="%.1fx", color=TEXTO, fontsize=9, padding=2)
    ax.set_xlabel("Hilos externos x internos")
    ax.set_ylabel("Speed-up frente al secuencial")
    ax.set_title(f"Paralelismo anidado: cómo repartir los hilos ({cpu})")
    ax.grid(axis="x", visible=False)
    if any(c == SERIES[1] for c in colores):
        from matplotlib.patches import Patch

        ax.legend(handles=[Patch(color=SERIES[0], label="Reparto del total de hilos"),
                           Patch(color=SERIES[1], label="Mismo número en los dos niveles")],
                  loc="upper left")
    guardar(fig, salida, "cpu_anidado.png")


def metodos(filas, salida, cpu):
    datos = [f for f in filas if f["experimento"] == "metodos"]
    if not datos:
        return
    nombres = {
        "directo_sec": "Directo\nsecuencial",
        "directo_omp": "Directo\nOpenMP",
        "por_terminos_sec": "Por términos\nsecuencial",
        "por_terminos_omp": "Por términos\nOpenMP",
    }
    etiquetas = [nombres.get(f["metodo"], f["metodo"]) for f in datos]
    ms = [1e3 * float(f["tiempo_s"]) for f in datos]
    fig, ax = plt.subplots(figsize=(7, 4))
    barras = ax.bar(etiquetas, ms, color=SERIES[0], width=0.6, edgecolor=FONDO, linewidth=2)
    ax.bar_label(barras, labels=[f"{v:.2f} ms" if v < 10 else f"{v:.0f} ms" for v in ms],
                 color=TEXTO, fontsize=9, padding=2)
    ax.set_yscale("log")
    ax.set_ylabel("Tiempo (ms, escala log)")
    ax.set_title(f"Método directo frente a método por términos ({cpu})")
    ax.grid(axis="x", visible=False)
    guardar(fig, salida, "cpu_metodos.png")


def gpu_bloques(filas, salida, gpu):
    datos = [f for f in filas if f["experimento"] == "base"]
    if not datos:
        return
    por_kernel = defaultdict(list)
    for f in datos:
        por_kernel[f["kernel"]].append((int(f["bloque"]), float(f["kernel_ms"])))
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (kernel, puntos) in enumerate(por_kernel.items()):
        puntos.sort()
        ax.plot([p[0] for p in puntos], [p[1] for p in puntos], marker="o", markersize=7,
                color=SERIES[i % len(SERIES)], label=kernel)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    bloques = sorted({int(f["bloque"]) for f in datos})
    ax.set_xticks(bloques, [str(b) for b in bloques])
    ax.set_xlabel("Hilos por bloque")
    ax.set_ylabel("Tiempo del kernel (ms, escala log)")
    ax.set_title(f"Kernels CUDA según el tamaño de bloque ({gpu})")
    ax.legend(loc="best")
    guardar(fig, salida, "gpu_bloques.png")


def gpu_desglose(filas, salida, gpu):
    """Reparto del tiempo de cada kernel (con su mejor bloque) entre copias y cálculo."""
    datos = [f for f in filas if f["experimento"] == "base"]
    if not datos:
        return
    mejores = {}
    for f in datos:
        k = f["kernel"]
        if k not in mejores or float(f["total_ms"]) < float(mejores[k]["total_ms"]):
            mejores[k] = f
    kernels = list(mejores)
    partes = [("h2d_ms", "Copia CPU → GPU"), ("kernel_ms", "Kernel"), ("d2h_ms", "Copia GPU → CPU")]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    izquierda = [0.0] * len(kernels)
    for i, (col, nombre) in enumerate(partes):
        pct = [100 * float(mejores[k][col]) / float(mejores[k]["total_ms"]) for k in kernels]
        ax.barh(kernels, pct, left=izquierda, color=SERIES[i], label=nombre, height=0.55,
                edgecolor=FONDO, linewidth=2)
        izquierda = [a + b for a, b in zip(izquierda, pct)]
    for y, k in enumerate(kernels):
        ax.text(101.5, y, f"{float(mejores[k]['total_ms']):.2f} ms", va="center", color=TEXTO, fontsize=9)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100], ["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.set_xlabel("Parte del tiempo total (mejor tamaño de bloque)")
    ax.set_title(f"En qué se va el tiempo en la GPU ({gpu})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.grid(axis="y", visible=False)
    guardar(fig, salida, "gpu_desglose.png")


def gpu_escalado(filas, salida, gpu):
    """Tiempo frente a número de variables: OpenMP con todos los hilos y los dos mejores kernels."""
    datos = [f for f in filas if f["experimento"] in ("base", "escalado_vars")]
    if not datos:
        return
    mejor = {}
    omp = {}
    for f in datos:
        vars_ = int(f["vars"])
        total = float(f["total_ms"])
        omp[vars_] = total * float(f["speedup_vs_omp"])
        clave = (f["kernel"], vars_)
        if clave not in mejor or total < mejor[clave]:
            mejor[clave] = total
    xs = sorted(omp)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, [omp[x] for x in xs], color=TEXTO_SUAVE, marker="o", markersize=7, label="CPU, OpenMP 16 hilos")
    for i, (kernel, nombre) in enumerate([("directo_original", "GPU, kernel original"),
                                          ("directo_tiled", "GPU, tiled"),
                                          ("por_terminos", "GPU, por términos")]):
        ys = [mejor.get((kernel, x)) for x in xs]
        ax.plot(xs, ys, color=SERIES[i], marker="o", markersize=7, label=nombre)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xs, [f"{x:,}".replace(",", ".") for x in xs])
    ax.set_xlabel("Variables del polinomio (20.000 términos)")
    ax.set_ylabel("Tiempo total (ms, escala log)")
    ax.set_title(f"CPU frente a GPU al crecer el problema ({gpu})")
    ax.legend(loc="upper left")
    guardar(fig, salida, "gpu_escalado.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu", type=Path)
    p.add_argument("--gpu", type=Path)
    p.add_argument("--nombre-cpu", default="Ryzen 7 7700X")
    p.add_argument("--nombre-gpu", default="RTX 5070")
    p.add_argument("--salida", type=Path, default=Path("docs/img"))
    a = p.parse_args()
    a.salida.mkdir(parents=True, exist_ok=True)
    estilo()
    print("Gráficas generadas:")
    if a.cpu:
        filas = leer(a.cpu)
        escalado(filas, a.salida, a.nombre_cpu)
        anidado(filas, a.salida, a.nombre_cpu)
        metodos(filas, a.salida, a.nombre_cpu)
    if a.gpu:
        filas = leer(a.gpu)
        gpu_bloques(filas, a.salida, a.nombre_gpu)
        gpu_desglose(filas, a.salida, a.nombre_gpu)
        gpu_escalado(filas, a.salida, a.nombre_gpu)


if __name__ == "__main__":
    main()
