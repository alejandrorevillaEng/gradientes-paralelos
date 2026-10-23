#ifndef GRADIENTE_H
#define GRADIENTE_H

#include "polinomio.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Paso por defecto de las diferencias finitas centradas. */
#define PASO_H_DEFECTO 1e-6

/*
 * Gradiente "caja negra" de cualquier f: R^n -> R por diferencias centradas:
 *   df/dx_i ~ (f(x + h e_i) - f(x - h e_i)) / 2h
 * Cuesta 2n evaluaciones de f.
 */
void gradiente_fd_sec(FuncionEscalar f, const void *ctx, const double *x, int n,
                      double h, double *grad);

/* Igual, repartiendo las variables entre hilos (cada hilo usa su copia de x). */
void gradiente_fd_omp(FuncionEscalar f, const void *ctx, const double *x, int n,
                      double h, double *grad, int hilos);

/* --- Versiones para polinomios ------------------------------------------- */

/* Método directo (el de la práctica): 2n evaluaciones del polinomio entero. */
void grad_directo_sec(const Polinomio *p, const double *x, double h, double *grad);

/* Paralelismo de un nivel: bucle de variables repartido entre hilos. */
void grad_directo_omp(const Polinomio *p, const double *x, double h, double *grad,
                      int hilos);

/*
 * Paralelismo anidado: hilos_ext reparten las variables y cada uno abre un
 * equipo de hilos_int para sumar los términos del polinomio (reducción).
 * En total se usan hilos_ext * hilos_int hilos.
 */
void grad_directo_anidado(const Polinomio *p, const double *x, double h, double *grad,
                          int hilos_ext, int hilos_int);

/*
 * Método por términos. Como f es una suma de términos, la diferencia centrada
 * de f es la suma de las diferencias centradas de cada término, y un término
 * que no contiene x_i aporta exactamente 0. Da el mismo resultado que el
 * método directo con O(términos) operaciones en vez de O(n * términos).
 */
void grad_por_terminos_sec(const Polinomio *p, const double *x, double h, double *grad);
void grad_por_terminos_omp(const Polinomio *p, const double *x, double h, double *grad,
                           int hilos);

/* Evaluación del polinomio con la suma repartida entre hilos (reducción). */
double polinomio_evaluar_omp(const Polinomio *p, const double *x, int hilos);

/* ||a - b||_2 / ||b||_2 (o ||a||_2 si b es el vector nulo). */
double error_relativo(const double *a, const double *b, int n);

#ifdef __cplusplus
}
#endif

#endif
