#ifndef CRONOMETRO_H
#define CRONOMETRO_H

/*
 * Reloj de alta resolución en segundos. omp_get_wtime() en MinGW solo tiene
 * resolución de milisegundos, que no basta para las versiones rápidas.
 */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static inline double cronometro_ahora(void) {
    static LARGE_INTEGER frecuencia = {0};
    LARGE_INTEGER t;
    if (frecuencia.QuadPart == 0) QueryPerformanceFrequency(&frecuencia);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)frecuencia.QuadPart;
}
#else
#include <time.h>
static inline double cronometro_ahora(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}
#endif

#endif
