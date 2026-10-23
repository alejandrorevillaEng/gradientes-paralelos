# Cómo contribuir

Es un proyecto personal, pero acepto issues y pull requests, sobre todo con medidas en otras CPU y GPU.

## Preparar el entorno

Hace falta CMake 3.25 o superior y GCC o Clang con OpenMP. Para la parte de GPU, el CUDA Toolkit.

```bash
cmake --preset release && cmake --build --preset release && ctest --preset release
cmake --preset sanitizers && cmake --build --preset sanitizers && ctest --preset sanitizers
```

## Antes de abrir un pull request

- `ctest` pasa en release y con sanitizers, y el código compila sin avisos con GCC y con Clang.
- Si añades una función a la biblioteca, añade su test en `tests/test_main.c` y, si tiene sentido desde Python, en `python/tests/`.
- Si cambias un kernel, compara su resultado con `polinomio_gradiente_exacto` y apunta el error relativo.
- Los números del README salen de un CSV de `resultados/`. Si cambian, sube también el CSV y vuelve a generar las gráficas y la página:

  ```bash
  python bench/graficas.py --cpu resultados/... --gpu resultados/...
  python bench/pagina.py --cpu resultados/... --gpu resultados/... --salida docs/datos.js
  ```

- Anota el cambio en `CHANGELOG.md`.

## Aportar medidas de otro equipo

```bash
./build/release/gradbench --csv resultados/cpu_<modelo>.csv
./build/cuda/gradbench_cuda --csv resultados/gpu_<modelo>.csv
```

Indica en el pull request el sistema operativo, el compilador y su versión.

## Commits

Mensajes cortos en español, en imperativo y con un prefijo: `feat:`, `fix:`, `perf:`, `test:`, `docs:` o `build:`.
