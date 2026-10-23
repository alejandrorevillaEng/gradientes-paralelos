# De la práctica entregada a este repositorio

Este proyecto parte de la práctica final de Programación Paralela (Grado en Ingeniería Informática, UCAM, curso 2025-26). Al repasarla meses después para publicarla encontré fallos en el código y en las conclusiones. Los dejo apuntados porque son lo que más aprendí rehaciéndola.

## Lo que pedía el enunciado y cómo queda

| Requisito | Entrega original | Ahora |
|---|---|---|
| Gradiente numérico de una función arbitraria | Solo polinomios | `gradiente_fd_sec/omp` acepta cualquier `f: R^n -> R` (hay un test con Rosenbrock) |
| OpenMP y análisis de escalabilidad | Hecho | Igual, medido con un reloj de alta resolución y la mediana de varias repeticiones |
| Estudio del paralelismo anidado repartiendo hilos entre bucles | Una sola configuración | Todos los repartos externo × interno con el mismo total de hilos |
| CUDA correcto frente al secuencial | Tolerancia de 100 en el error acumulado | Error relativo frente al gradiente analítico (≈1e-9) |
| Variar el tamaño de bloque | Solo en la memoria; el código tenía 128 fijo | Barrido de 32 a 1024 en todos los kernels |
| Desglose de tiempos CUDA | Hecho | Igual, con memoria fijada (pinned) |
| Parte opcional: optimizar el kernel | No | Tres kernels nuevos (ver README) |
| Extender a sistemas de ecuaciones | Solo un párrafo | Jacobiana en paralelo (OpenMP y CUDA) y Newton-Raphson; resuelve el sistema del enunciado |

## Fallos que encontré

**El código entregado no compilaba.** Al copiarlo a los `.txt` de la entrega, los `\n` de los `printf` se convirtieron en `%n` y la barra de continuación de la macro `CUDA_CHECK` en `%`. Un `printf` con `%n` y sin argumento es comportamiento indefinido: en el mejor caso no imprime bien y en el peor cuelga el programa.

**El speed-up de CUDA estaba inflado unas cinco veces.** El tiempo secuencial que usamos (2.773 ms) era el de la versión OpenMP, que repite el cálculo 5 veces. El de CUDA (15,96 ms) era una sola ejecución. Con las mismas medidas del laboratorio, pero comparando una ejecución con una ejecución:

| Comparación (RTX 2080 Ti, 500 variables, 20.000 términos) | Memoria | Corregido |
|---|---|---|
| CUDA frente al secuencial (509 ms / 15,96 ms) | 173,7x | ≈32x |
| CUDA frente a OpenMP con 12 hilos (242,9 / 5 = 48,6 ms) | 15,2x | ≈3x |

**El paralelismo anidado usaba más hilos de los que creíamos.** Con `omp_set_num_threads(4)` y el anidamiento activado, cada uno de los 4 hilos externos abre otro equipo de 4: en total 16 hilos. Por eso salía un speed-up "superlineal" de 8,63x con "4 hilos", que en la memoria atribuimos a la caché. Aquí se mide el reparto real (1×16, 2×8, 4×4, 8×2, 16×1) y el caso de 16×16 hilos.

**La conclusión sobre el anidamiento era la contraria.** Abrir un equipo de hilos para cada evaluación del polinomio (1.000 por gradiente) cuesta más de lo que se gana: todos los repartos con hilos internos van peor que paralelizar solo el bucle externo.

**El kernel restaba dos números enormes.** Sumaba f(x+h) y f(x−h) sobre los 20.000 términos y luego los restaba, perdiendo cifras significativas. Como f es una suma de términos, basta con sumar la diferencia de los términos que contienen la variable: mismo resultado, más preciso y con menos trabajo.

**Otras cosas.** La memoria decía que se usaba memoria fijada y el código entregado usaba `malloc`. El generador con `rand()` no da los mismos números en todos los sistemas; ahora hay uno propio (splitmix64). Faltaban la valoración final del uso de la IA y referencias reales en la bibliografía.
