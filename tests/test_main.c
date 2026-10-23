/*
 * Tests de las versiones de CPU. Sin dependencias: cada test es una función
 * y COMPROBAR registra el fallo con fichero y línea.
 */
#include "gradiente.h"
#include "polinomio.h"
#include "sistema.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fallos = 0;
static int comprobaciones = 0;

#define COMPROBAR(cond)                                                     \
    do {                                                                    \
        comprobaciones++;                                                   \
        if (!(cond)) {                                                      \
            fallos++;                                                       \
            fprintf(stderr, "  FALLO %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                   \
    } while (0)

#define CERCA(a, b, tol) COMPROBAR(fabs((a) - (b)) <= (tol))

/* f(x, y, z) = 3xy + 4yz + 2x + 5, el ejemplo del enunciado. */
static void polinomio_enunciado(Polinomio *p) {
    polinomio_crear(p, 3, 4);
    polinomio_agregar_termino(p, 0, 3.0, 2, (int[]){0, 1}, (int[]){1, 1});
    polinomio_agregar_termino(p, 1, 4.0, 2, (int[]){1, 2}, (int[]){1, 1});
    polinomio_agregar_termino(p, 2, 2.0, 1, (int[]){0}, (int[]){1});
    polinomio_agregar_termino(p, 3, 5.0, 0, NULL, NULL);
}

static void test_ejemplo_enunciado(void) {
    Polinomio p;
    polinomio_enunciado(&p);
    const double x[3] = {1.0, 2.0, 3.0};
    double g[3];

    CERCA(polinomio_evaluar(&p, x), 6.0 + 24.0 + 2.0 + 5.0, 1e-12);

    polinomio_gradiente_exacto(&p, x, g);
    CERCA(g[0], 8.0, 1e-12);
    CERCA(g[1], 15.0, 1e-12);
    CERCA(g[2], 8.0, 1e-12);

    grad_directo_sec(&p, x, PASO_H_DEFECTO, g);
    CERCA(g[0], 8.0, 1e-6);
    CERCA(g[1], 15.0, 1e-6);
    CERCA(g[2], 8.0, 1e-6);
    polinomio_liberar(&p);
}

static void test_exponentes(void) {
    /* f = 2 x^3 y^2 -> df/dx = 6 x^2 y^2, df/dy = 4 x^3 y */
    Polinomio p;
    polinomio_crear(&p, 2, 1);
    polinomio_agregar_termino(&p, 0, 2.0, 2, (int[]){0, 1}, (int[]){3, 2});
    const double x[2] = {1.5, -2.0};
    double g[2];
    polinomio_gradiente_exacto(&p, x, g);
    CERCA(g[0], 6.0 * 2.25 * 4.0, 1e-12);
    CERCA(g[1], 4.0 * 3.375 * -2.0, 1e-12);
    grad_por_terminos_sec(&p, x, 1e-5, g);
    CERCA(g[0], 54.0, 1e-6);
    CERCA(g[1], -27.0, 1e-6);
    polinomio_liberar(&p);
}

static double rosenbrock(const double *x, const void *ctx) {
    const int n = *(const int *)ctx;
    double s = 0.0;
    for (int i = 0; i < n - 1; i++) {
        const double a = x[i + 1] - x[i] * x[i], b = 1.0 - x[i];
        s += 100.0 * a * a + b * b;
    }
    return s;
}

static void test_funcion_arbitraria(void) {
    /* El gradiente "caja negra" funciona con cualquier función, no solo con polinomios. */
    int n = 6;
    double x[6] = {-1.2, 1.0, 0.5, -0.3, 2.0, 1.1}, g[6], g_omp[6];
    gradiente_fd_sec(rosenbrock, &n, x, n, 1e-6, g);
    gradiente_fd_omp(rosenbrock, &n, x, n, 1e-6, g_omp, 4);
    for (int i = 0; i < n; i++) {
        double exacto = 0.0;
        if (i < n - 1) exacto += -400.0 * x[i] * (x[i + 1] - x[i] * x[i]) - 2.0 * (1.0 - x[i]);
        if (i > 0) exacto += 200.0 * (x[i] - x[i - 1] * x[i - 1]);
        CERCA(g[i], exacto, 1e-4 * (1.0 + fabs(exacto)));
        CERCA(g_omp[i], g[i], 0.0);
    }
}

static void test_versiones_coinciden(void) {
    Rng rng;
    rng_iniciar(&rng, 7);
    Polinomio p;
    const int n = 120;
    COMPROBAR(polinomio_aleatorio(&p, n, 3000, 3, 2, 10.0, &rng) == 0);
    double *x = malloc(n * sizeof(double)), *ex = malloc(n * sizeof(double));
    double *g = malloc(n * sizeof(double));
    punto_aleatorio(x, n, 2.0, &rng);
    polinomio_gradiente_exacto(&p, x, ex);

    grad_directo_sec(&p, x, PASO_H_DEFECTO, g);
    COMPROBAR(error_relativo(g, ex, n) < 1e-7);
    grad_directo_omp(&p, x, PASO_H_DEFECTO, g, 4);
    COMPROBAR(error_relativo(g, ex, n) < 1e-7);
    grad_directo_anidado(&p, x, PASO_H_DEFECTO, g, 2, 2);
    COMPROBAR(error_relativo(g, ex, n) < 1e-7);
    grad_directo_anidado(&p, x, PASO_H_DEFECTO, g, 3, 1);
    COMPROBAR(error_relativo(g, ex, n) < 1e-7);
    grad_por_terminos_sec(&p, x, PASO_H_DEFECTO, g);
    COMPROBAR(error_relativo(g, ex, n) < 1e-7);
    grad_por_terminos_omp(&p, x, PASO_H_DEFECTO, g, 4);
    COMPROBAR(error_relativo(g, ex, n) < 1e-7);
    CERCA(polinomio_evaluar_omp(&p, x, 4), polinomio_evaluar(&p, x),
          1e-9 * fabs(polinomio_evaluar(&p, x)));

    free(x);
    free(ex);
    free(g);
    polinomio_liberar(&p);
}

static void test_generador_reproducible(void) {
    Rng a, b;
    rng_iniciar(&a, 42);
    rng_iniciar(&b, 42);
    for (int i = 0; i < 100; i++) COMPROBAR(rng_siguiente(&a) == rng_siguiente(&b));
    rng_iniciar(&a, 1);
    for (int i = 0; i < 1000; i++) {
        double u = rng_uniforme(&a, 3.0);
        int k = rng_entero(&a, 2, 5);
        COMPROBAR(u >= -3.0 && u < 3.0);
        COMPROBAR(k >= 2 && k <= 5);
    }
    /* Las variables de un término aleatorio no se repiten. */
    Polinomio p;
    rng_iniciar(&a, 3);
    polinomio_aleatorio(&p, 5, 500, 3, 2, 1.0, &a);
    for (int t = 0; t < p.num_terms; t++) {
        for (int i = 0; i < p.num_factores[t]; i++) {
            for (int j = i + 1; j < p.num_factores[t]; j++) {
                COMPROBAR(p.var[t * MAX_FACTORES + i] != p.var[t * MAX_FACTORES + j]);
            }
        }
    }
    polinomio_liberar(&p);
}

static void test_resolver_lineal(void) {
    double A[9] = {0, 2, 1, 1, 1, 1, 2, 1, 3};  /* el 0 obliga a pivotar */
    double b[3] = {7, 6, 13}, x[3];  /* b = A * (1, 2, 3) */
    COMPROBAR(resolver_lineal(3, A, b, x, 2) == 0);
    CERCA(x[0], 1.0, 1e-12);
    CERCA(x[1], 2.0, 1e-12);
    CERCA(x[2], 3.0, 1e-12);

    double S[4] = {1, 2, 2, 4}, c[2] = {1, 2}, y[2];
    COMPROBAR(resolver_lineal(2, S, c, y, 1) == -1);
}

static void test_sistema_enunciado(void) {
    Sistema s;
    COMPROBAR(sistema_ejemplo_enunciado(&s) == 0);
    double x[3] = {0.0, 0.0, -3.0}, F[3], J[9], Jex[9];
    jacobiana_directa_sec(&s, x, PASO_H_DEFECTO, J);
    jacobiana_exacta(&s, x, Jex);
    COMPROBAR(error_relativo(J, Jex, 9) < 1e-8);

    ResultadoNewton r = newton_resolver(&s, x, 100, 1e-12, PASO_H_DEFECTO, 1);
    COMPROBAR(r.convergio);
    sistema_residuo(&s, x, F);
    for (int i = 0; i < 3; i++) CERCA(F[i], 0.0, 1e-10);
    /* Es una de las dos soluciones reales del sistema. */
    const int sol_a = fabs(x[0] - 0.19171424) < 1e-6 && fabs(x[1] - 0.13415774) < 1e-6 &&
                      fabs(x[2] + 2.74235492) < 1e-6;
    const int sol_b = fabs(x[0] - 0.41005988) < 1e-6 && fabs(x[1] - 0.11828467) < 1e-6 &&
                      fabs(x[2] + 3.13377031) < 1e-6;
    COMPROBAR(sol_a || sol_b);
    sistema_liberar(&s);
}

static void test_newton_sistema_aleatorio(void) {
    Rng rng;
    rng_iniciar(&rng, 11);
    const int n = 40;
    Sistema s;
    double sol[40], x[40], J[1600], Jd[1600], Jex[1600];
    COMPROBAR(sistema_aleatorio(&s, n, 8, &rng, sol) == 0);
    for (int i = 0; i < n; i++) x[i] = sol[i] + rng_uniforme(&rng, 0.3);

    jacobiana_exacta(&s, x, Jex);
    jacobiana_directa_omp(&s, x, PASO_H_DEFECTO, Jd, 4);
    jacobiana_por_terminos_omp(&s, x, PASO_H_DEFECTO, J, 4);
    COMPROBAR(error_relativo(Jd, Jex, n * n) < 1e-8);
    COMPROBAR(error_relativo(J, Jex, n * n) < 1e-8);

    ResultadoNewton r = newton_resolver(&s, x, 50, 1e-11, PASO_H_DEFECTO, 4);
    COMPROBAR(r.convergio);
    COMPROBAR(r.iteraciones <= 10);
    COMPROBAR(error_relativo(x, sol, n) < 1e-9);
    sistema_liberar(&s);
}

static void test_terminos_invalidos(void) {
    Polinomio p;
    polinomio_crear(&p, 2, 2);
    COMPROBAR(polinomio_agregar_termino(&p, 0, 1.0, 1, (int[]){2}, (int[]){1}) == -1);
    COMPROBAR(polinomio_agregar_termino(&p, 0, 1.0, 1, (int[]){-1}, (int[]){1}) == -1);
    COMPROBAR(polinomio_agregar_termino(&p, 0, 1.0, 1, (int[]){0}, (int[]){0}) == -1);
    COMPROBAR(polinomio_agregar_termino(&p, 2, 1.0, 1, (int[]){0}, (int[]){1}) == -1);
    COMPROBAR(polinomio_agregar_termino(&p, 0, 1.0, MAX_FACTORES + 1, NULL, NULL) == -1);
    COMPROBAR(polinomio_agregar_termino(&p, 1, 1.0, 2, (int[]){0, 1}, (int[]){1, 2}) == 1);
    polinomio_liberar(&p);
}

int main(void) {
    struct {
        const char *nombre;
        void (*fn)(void);
    } tests[] = {
        {"ejemplo del enunciado", test_ejemplo_enunciado},
        {"exponentes", test_exponentes},
        {"funcion arbitraria (Rosenbrock)", test_funcion_arbitraria},
        {"todas las versiones coinciden con el gradiente exacto", test_versiones_coinciden},
        {"generador reproducible", test_generador_reproducible},
        {"eliminacion gaussiana", test_resolver_lineal},
        {"sistema del enunciado con Newton", test_sistema_enunciado},
        {"Newton en sistema aleatorio de 40 ecuaciones", test_newton_sistema_aleatorio},
        {"terminos con variables o exponentes fuera de rango", test_terminos_invalidos},
    };
    const int n = (int)(sizeof(tests) / sizeof(tests[0]));
    for (int i = 0; i < n; i++) {
        const int antes = fallos;
        tests[i].fn();
        printf("[%s] %s\n", fallos == antes ? " OK " : "FALLO", tests[i].nombre);
    }
    printf("\n%d tests, %d comprobaciones, %d fallos\n", n, comprobaciones, fallos);
    return fallos ? 1 : 0;
}
