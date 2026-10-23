# Gradientes en paralelo: C, OpenMP y CUDA

Cálculo numérico del gradiente de funciones multivariables, y de la jacobiana de sistemas de ecuaciones, en versión secuencial, OpenMP y CUDA, con benchmarks reproducibles y validación contra la solución analítica.

[![CI](https://github.com/alejandrorevillaEng/gradientes-paralelos/actions/workflows/ci.yml/badge.svg)](https://github.com/alejandrorevillaEng/gradientes-paralelos/actions/workflows/ci.yml)
[![Seguridad](https://github.com/alejandrorevillaEng/gradientes-paralelos/actions/workflows/seguridad.yml/badge.svg)](https://github.com/alejandrorevillaEng/gradientes-paralelos/actions/workflows/seguridad.yml)
[![Licencia: MIT](https://img.shields.io/badge/licencia-MIT-blue.svg)](LICENSE)

![Escalabilidad con el número de hilos](docs/img/cpu_escalado.png)

## Por qué existe

Es la práctica final de Programación Paralela (Grado en Ingeniería Informática, UCAM, curso 2025-26). Calcular gradientes es la base del entrenamiento de redes neuronales y de casi cualquier método de optimización, y quería entender de verdad cuánto se gana repartiendo ese cálculo entre núcleos de CPU y miles de hilos de GPU. Al rehacerla para publicarla encontré errores en la versión entregada (el código no compilaba y el speed-up de CUDA estaba inflado unas cinco veces); los cuento en [docs/practica-original.md](docs/practica-original.md).

## Qué hace

- Gradiente por diferencias centradas de cualquier función `f: R^n -> R` (secuencial y OpenMP) y versiones específicas para polinomios de miles de términos.
- Dos formas de calcularlo: la directa, que evalúa el polinomio entero 2n veces, y una por términos, que aprovecha que f es una suma y solo toca los términos que contienen cada variable.
- Paralelismo de un nivel y anidado, con todos los repartos de hilos entre el bucle externo y el interno.
- Cuatro kernels CUDA, con memoria fijada, desglose de tiempos (copias y kernel) y barrido de tamaños de bloque.
- Sistemas de ecuaciones polinómicas: jacobiana en paralelo y Newton-Raphson amortiguado con eliminación gaussiana. Resuelve el sistema de ejemplo del enunciado.
- Validación de cada resultado frente al gradiente analítico y CSV con todas las medidas.

## Stack y decisiones técnicas

| Pieza | Tecnología | Por qué |
|---|---|---|
| Cálculo | C11 | Lo pedía la asignatura y deja ver el coste real de cada acceso a memoria |
| CPU | OpenMP | Paraleliza bucles con directivas; la reducción de arrays (`reduction(+:grad[0:n])`) evita los `atomic` en el método por términos |
| GPU | CUDA (C++17) | Los polinomios van en arrays planos para copiarlos con un `cudaMemcpy` por array y que los hilos lean posiciones contiguas |
| Compilación | CMake | Detecta OpenMP y, si hay `nvcc`, compila también la versión CUDA para la GPU de la máquina |
| Gráficas | Python y matplotlib | Se generan siempre a partir de los CSV, así cualquiera puede repetirlas con sus propios datos |

Los datos de prueba salen de un generador propio (splitmix64) con semilla fija, porque `rand()` no da la misma secuencia en Linux y en Windows. Los tiempos usan un reloj de alta resolución y la mediana de varias repeticiones; si una ejecución dura menos de 50 ms se repite en bucle para no medir solo el ruido del reloj.

## Resultados en CPU

Medidos en un AMD Ryzen 7 7700X (8 núcleos, 16 hilos) con GCC 16 en Windows, 500 variables y 20.000 términos, mediana de 5 repeticiones. Los datos están en [resultados/](resultados/).

| Hilos | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Tiempo (ms) | 245 | 127 | 66 | 37 | 26 |
| Speed-up | 0,96x | 1,86x | 3,55x | 6,35x | 8,92x |

Escala casi linealmente hasta los 8 núcleos físicos. De 8 a 16 hilos gana bastante menos (de 6,4x a 8,9x) porque los hilos 9 a 16 comparten núcleo con los primeros (SMT). Con problemas más grandes el speed-up con 16 hilos sube de 9,5x (20.000 términos) a 11,1x (160.000), porque el coste de crear y sincronizar los hilos pesa menos.

![Repartos de hilos en el paralelismo anidado](docs/img/cpu_anidado.png)

El paralelismo anidado no compensa en este problema. Cada hilo externo abre un equipo nuevo para cada una de las 1.000 evaluaciones del polinomio, y eso cuesta más que lo que se gana repartiendo la suma. Cuantos más hilos internos, peor: 16×1 da 9,3x y 1×16 da 0,17x (más lento que el secuencial). Usar 16 hilos en cada nivel (256 en total, lo que pasaba sin querer en la versión original) se queda en 0,26x.

![Método directo frente a método por términos](docs/img/cpu_metodos.png)

El cambio de algoritmo gana mucho más que el paralelismo: el método por términos tarda 0,51 ms en secuencial, unas 470 veces menos que el directo, porque cada término tiene como mucho 3 variables y solo se recalcula donde cambia algo. Con 16 hilos baja a 0,15 ms. Además es más preciso (error relativo 9e-11 frente a 2e-9), porque no resta dos sumas enormes.

En sistemas de 200 ecuaciones con 100 términos por ecuación, la jacobiana tarda 21,3 ms en secuencial, 3,7 ms con OpenMP y 0,15 ms por términos. Newton converge en 3 iteraciones con un residuo de 1e-13.

## Resultados en GPU

La parte de CUDA se ejecuta en cualquier equipo con una GPU NVIDIA (CMake detecta su arquitectura) o en Google Colab con [notebooks/benchmark_cuda_colab.ipynb](notebooks/benchmark_cuda_colab.ipynb), que clona el repositorio, compila, pasa los tests y genera el CSV y las gráficas. En GitHub Actions solo se comprueba que compila, porque los runners no tienen GPU.

Los kernels que compara:

| Kernel | Idea |
|---|---|
| `directo_original` | El de la práctica: un hilo por variable que suma f(x+h) y f(x−h) sobre todos los términos |
| `directo` | Igual, pero solo suma la diferencia de los términos que contienen la variable |
| `directo_tiled` | El anterior cargando los términos por bloques en memoria compartida |
| `por_terminos` | Un hilo por término que suma su aportación con `atomicAdd`; también calcula jacobianas |

Como referencia, la versión original medida en el laboratorio de la universidad (RTX 2080 Ti) tardaba 15,96 ms en total (0,19 ms de copia a la GPU, 15,76 ms de kernel y 0,02 ms de vuelta) frente a 509 ms del secuencial: unas 32 veces más rápida. Con 500 variables solo hay 500 hilos, muy pocos para llenar una GPU, y por eso los bloques pequeños (128) iban mejor que los de 1.024, que dejan casi todos los multiprocesadores parados.

## Cómo ejecutarlo

Requisitos: CMake 3.24 o superior y un compilador de C con OpenMP (GCC o Clang con libomp). CUDA es opcional.

```bash
git clone https://github.com/alejandrorevillaEng/gradientes-paralelos.git
cd gradientes-paralelos
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

```bash
./build/gradbench --experimento todo --csv resultados/mi_cpu.csv
./build/sistema_demo
./build/gradbench_cuda --csv resultados/mi_gpu.csv     # solo si se ha compilado con CUDA
```

`gradbench --help` lista las opciones (variables, términos, lista de hilos, repeticiones, semilla). Para las gráficas:

```bash
pip install -r bench/requirements.txt
python bench/graficas.py --cpu resultados/mi_cpu.csv --gpu resultados/mi_gpu.csv
```

## Tests

```bash
ctest --test-dir build --output-on-failure
```

Comprueban el ejemplo del enunciado (∇f(1, 2, 3) = (8, 15, 8)), polinomios con exponentes, una función que no es un polinomio (Rosenbrock), que todas las versiones paralelas coinciden con el gradiente analítico, la eliminación gaussiana con pivoteo y que Newton resuelve el sistema del enunciado y uno aleatorio de 40 ecuaciones. En CI se compila con GCC y con Clang y los avisos cuentan como errores.

## Estructura

```
include/ y src/   biblioteca: polinomios, gradientes, sistemas y Newton
src/main_*.c      gradbench (benchmarks) y sistema_demo
cuda/             kernels y benchmark de GPU
tests/            tests de la parte de CPU
bench/            gráficas a partir de los CSV
notebooks/        benchmark de CUDA en Google Colab
resultados/       CSV y salida de las mediciones
```

## Autor

Alejandro Revilla Blanco

## Licencia

[MIT](LICENSE)
