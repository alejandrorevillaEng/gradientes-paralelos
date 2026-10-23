#ifndef SISTEMA_H
#define SISTEMA_H

#include "polinomio.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Sistema de ecuaciones polinómicas F(x) = 0 con
 *   F_i(x) = ecs[i](x) - rhs[i],  i = 0..num_ecs-1
 * La matriz jacobiana tiene en la fila i el gradiente de la ecuación i.
 */
typedef struct {
    int num_ecs;
    int num_vars;
    Polinomio *ecs;
    double *rhs;
} Sistema;

int sistema_crear(Sistema *s, int num_ecs, int num_vars);
void sistema_liberar(Sistema *s);

void sistema_residuo(const Sistema *s, const double *x, double *F);

/* Jacobianas (row-major, num_ecs x num_vars). */
void jacobiana_exacta(const Sistema *s, const double *x, double *J);
void jacobiana_directa_sec(const Sistema *s, const double *x, double h, double *J);
/* Reparte las num_ecs * num_vars derivadas parciales entre hilos. */
void jacobiana_directa_omp(const Sistema *s, const double *x, double h, double *J, int hilos);
/* Una fila por tarea, cada fila calculada por términos. */
void jacobiana_por_terminos_omp(const Sistema *s, const double *x, double h, double *J,
                                int hilos);

/* Resuelve A x = b (n x n) por eliminación gaussiana con pivoteo parcial.
 * A y b se modifican. Devuelve 0 si va bien y -1 si la matriz es singular. */
int resolver_lineal(int n, double *A, double *b, double *x, int hilos);

typedef struct {
    int iteraciones;
    double residuo;  /* ||F(x)||_inf al terminar */
    int convergio;
} ResultadoNewton;

/*
 * Newton-Raphson amortiguado: en cada paso resuelve J dx = -F y, si el
 * residuo empeora, reduce el paso a la mitad (hasta 20 veces).
 */
ResultadoNewton newton_resolver(const Sistema *s, double *x, int max_iter, double tol,
                                double h, int hilos);

/* El ejemplo del enunciado (variables x, y, z):
 *   3x^2 + yz + z + 5 = 2
 *   2x + yz^3 + 2x + 2 = 0
 *   3xz + 8yz + z^2 + 1 = 4                                               */
int sistema_ejemplo_enunciado(Sistema *s);

/*
 * Sistema cuadrado aleatorio con solución conocida x_sol. Cada ecuación i
 * lleva además un término lineal fuerte en x_i para que la jacobiana no sea
 * singular y Newton converja desde un punto cercano.
 */
int sistema_aleatorio(Sistema *s, int n, int terms_por_ec, Rng *rng, double *x_sol);

#ifdef __cplusplus
}
#endif

#endif
