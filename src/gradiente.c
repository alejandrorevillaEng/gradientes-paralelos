#include "gradiente.h"

#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

void gradiente_fd_sec(FuncionEscalar f, const void *ctx, const double *x, int n,
                      double h, double *grad) {
    double *tmp = malloc((size_t)n * sizeof(double));
    if (!tmp) abort();
    memcpy(tmp, x, (size_t)n * sizeof(double));

    for (int i = 0; i < n; i++) {
        const double original = tmp[i];
        tmp[i] = original + h;
        const double f_mas = f(tmp, ctx);
        tmp[i] = original - h;
        const double f_menos = f(tmp, ctx);
        tmp[i] = original;  /* restaurar antes de pasar a la siguiente variable */
        grad[i] = (f_mas - f_menos) / (2.0 * h);
    }
    free(tmp);
}

void gradiente_fd_omp(FuncionEscalar f, const void *ctx, const double *x, int n,
                      double h, double *grad, int hilos) {
    #pragma omp parallel num_threads(hilos)
    {
        /* Copia privada del punto: si los hilos tocaran el mismo vector habría
         * condiciones de carrera. */
        double *tmp = malloc((size_t)n * sizeof(double));
        if (!tmp) abort();
        memcpy(tmp, x, (size_t)n * sizeof(double));

        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            const double original = tmp[i];
            tmp[i] = original + h;
            const double f_mas = f(tmp, ctx);
            tmp[i] = original - h;
            const double f_menos = f(tmp, ctx);
            tmp[i] = original;
            grad[i] = (f_mas - f_menos) / (2.0 * h);
        }
        free(tmp);
    }
}

void grad_directo_sec(const Polinomio *p, const double *x, double h, double *grad) {
    gradiente_fd_sec(polinomio_como_funcion, p, x, p->num_vars, h, grad);
}

void grad_directo_omp(const Polinomio *p, const double *x, double h, double *grad,
                      int hilos) {
    gradiente_fd_omp(polinomio_como_funcion, p, x, p->num_vars, h, grad, hilos);
}

double polinomio_evaluar_omp(const Polinomio *p, const double *x, int hilos) {
    double suma = 0.0;
    #pragma omp parallel for num_threads(hilos) reduction(+ : suma) schedule(static)
    for (int t = 0; t < p->num_terms; t++) {
        suma += polinomio_evaluar_termino(p, t, x);
    }
    return suma;
}

void grad_directo_anidado(const Polinomio *p, const double *x, double h, double *grad,
                          int hilos_ext, int hilos_int) {
    const int n = p->num_vars;
    const int niveles_previos = omp_get_max_active_levels();
    omp_set_max_active_levels(2);

    #pragma omp parallel num_threads(hilos_ext)
    {
        double *tmp = malloc((size_t)n * sizeof(double));
        if (!tmp) abort();
        memcpy(tmp, x, (size_t)n * sizeof(double));

        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            const double original = tmp[i];
            tmp[i] = original + h;
            const double f_mas = polinomio_evaluar_omp(p, tmp, hilos_int);
            tmp[i] = original - h;
            const double f_menos = polinomio_evaluar_omp(p, tmp, hilos_int);
            tmp[i] = original;
            grad[i] = (f_mas - f_menos) / (2.0 * h);
        }
        free(tmp);
    }
    omp_set_max_active_levels(niveles_previos);
}

/* Valor del término t cambiando solo la variable v por el valor dado. */
static double termino_con(const Polinomio *p, int t, const double *x, int v, double valor) {
    double r = p->coef[t];
    const int base = t * MAX_FACTORES;
    for (int k = 0; k < p->num_factores[t]; k++) {
        const int var = p->var[base + k];
        const double xv = var == v ? valor : x[var];
        for (int e = 0; e < p->exp[base + k]; e++) r *= xv;
    }
    return r;
}

/* Suma en grad la diferencia centrada del término t para cada una de sus variables. */
static void acumular_termino(const Polinomio *p, int t, const double *x, double h,
                             double *grad) {
    const int base = t * MAX_FACTORES;
    for (int k = 0; k < p->num_factores[t]; k++) {
        const int v = p->var[base + k];
        const double mas = termino_con(p, t, x, v, x[v] + h);
        const double menos = termino_con(p, t, x, v, x[v] - h);
        grad[v] += (mas - menos) / (2.0 * h);
    }
}

void grad_por_terminos_sec(const Polinomio *p, const double *x, double h, double *grad) {
    memset(grad, 0, (size_t)p->num_vars * sizeof(double));
    for (int t = 0; t < p->num_terms; t++) acumular_termino(p, t, x, h, grad);
}

void grad_por_terminos_omp(const Polinomio *p, const double *x, double h, double *grad,
                           int hilos) {
    const int n = p->num_vars;
    memset(grad, 0, (size_t)n * sizeof(double));
    /* Reducción de array (OpenMP 4.5): cada hilo acumula en una copia privada
     * de grad y al final se suman todas. */
    #pragma omp parallel for num_threads(hilos) reduction(+ : grad[0:n]) schedule(static)
    for (int t = 0; t < p->num_terms; t++) {
        acumular_termino(p, t, x, h, grad);
    }
}

double error_relativo(const double *a, const double *b, int n) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; i++) {
        const double d = a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    return den > 0.0 ? sqrt(num / den) : sqrt(num);
}
