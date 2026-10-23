"""Convierte los CSV de resultados en docs/datos.js para la página de GitHub Pages.

    python bench/pagina.py --cpu resultados/cpu_ryzen7700x.csv --gpu resultados/gpu_rtx5070.csv

La página (docs/index.html) no tiene datos escritos a mano: todo sale de aquí.
"""

import argparse
import csv
import json
from pathlib import Path


def leer(ruta):
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def numeros(fila):
    salida = {}
    for clave, valor in fila.items():
        try:
            salida[clave] = float(valor) if any(c in valor for c in ".e") else int(valor)
        except ValueError:
            salida[clave] = valor
    return salida


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu", type=Path, required=True)
    p.add_argument("--gpu", type=Path, required=True)
    p.add_argument("--nombre-cpu", default="AMD Ryzen 7 7700X (8 núcleos, 16 hilos)")
    p.add_argument("--nombre-gpu", default="NVIDIA GeForce RTX 5070")
    p.add_argument("--salida", type=Path, default=Path("docs/datos.js"))
    a = p.parse_args()

    datos = {
        "cpu": [numeros(f) for f in leer(a.cpu)],
        "gpu": [numeros(f) for f in leer(a.gpu)],
        "equipo": {"cpu": a.nombre_cpu, "gpu": a.nombre_gpu},
    }
    a.salida.write_text("window.DATOS = " + json.dumps(datos, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"{a.salida}: {len(datos['cpu'])} filas de CPU y {len(datos['gpu'])} de GPU")


if __name__ == "__main__":
    main()
