# Cambios

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/). Las versiones siguen [SemVer](https://semver.org/lang/es/).

## [1.1.0] - 2026-10-23

### Añadido
- Bindings de Python con `ctypes` (`python/gradientes`): polinomios, gradiente por los tres métodos con o sin hilos y gradiente "caja negra" de cualquier función de Python.
- Experimento `precision` en `gradbench_cuda`: el kernel directo en FP64 y en FP32 con dos pasos h. Resultados de la RTX 5070 en el README y en la página.
- `CMakePresets.json` con configuraciones de release, debug, sanitizers, cobertura y CUDA.
- Dockerfile para repetir el benchmark de CPU en cualquier máquina.
- CI: AddressSanitizer y UBSan, cppcheck, cobertura con mínimo del 85 % de líneas e imagen Docker.
- Workflow de release que publica los binarios de Linux y la biblioteca para Python.
- Documentación de arquitectura y decisiones, guía de contribución y este fichero.

### Cambiado
- `polinomio_agregar_termino` rechaza variables fuera de rango y exponentes menores que 1. Antes un índice malo hacía leer fuera del vector al evaluar.
- La medida de tiempos inicializa el array de repeticiones (aviso de cppcheck).

## [1.0.0] - 2026-10-23

Primera versión publicada: la práctica rehecha a partir de la entrega.

### Añadido
- Gradiente secuencial, OpenMP (uno y dos niveles) y método por términos.
- Cuatro kernels CUDA con desglose de tiempos y barrido de tamaños de bloque.
- Jacobiana, eliminación gaussiana y Newton-Raphson para sistemas polinómicos.
- Resultados en un Ryzen 7 7700X y una RTX 5070, gráficas y página interactiva.

### Corregido
- El código entregado no compilaba y el speed-up de CUDA estaba inflado unas cinco veces. Detalles en `docs/practica-original.md`.

[1.1.0]: https://github.com/alejandrorevillaEng/gradientes-paralelos/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/alejandrorevillaEng/gradientes-paralelos/releases/tag/v1.0.0
