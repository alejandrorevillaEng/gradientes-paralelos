#include "sistema.h"

#include "gradiente.h"

#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

int sistema_crear(Sistema *s, int num_ecs, int num_vars) {
    memset(s, 0, sizeof(*s));
    if (num_ecs <= 0 || num_vars <= 0) return -1;
    s->num_ecs = num_ecs;
    s->num_vars = num_vars;
    s->ecs = calloc((size_t)num_ecs, sizeof(Polinomio));
    s->rhs = calloc((size_t)num_ecs, sizeof(double));
    if (!s->ecs || !s->rhs) {
        sistema_liberar(s);
        return -1;
    }
    return 0;
}

void sistema_liberar(Sistema *s) {
    if (s->ecs) {
        for (int i = 0; i < s->num_ecs; i++) polinomio_liberar(&s->ecs[i]);
    }
    free(s->ecs);
    free(s->rhs);
    memset(s, 0, sizeof(*s));
}

void sistema_residuo(const Sistema *s, const double *x, double *F) {
    for (int i = 0; i < s->num_ecs; i++) F[i] = polinomio_evaluar(&s->ecs[i], x) - s->rhs[i];
}

void jacobiana_exacta(const Sistema *s, const double *x, double *J) {
    for (int i = 0; i < s->num_ecs; i++) {
        polinomio_gradiente_exacto(&s->ecs[i], x, J + (size_t)i * s->num_vars);
    }
}

void jacobiana_directa_sec(const Sistema *s, const double *x, double h, double *J) {
    for (int i = 0; i < s->num_ecs; i++) {
        grad_directo_sec(&s->ecs[i], x, h, J + (size_t)i * s->num_vars);
    }
}

void jacobiana_directa_omp(const Sistema *s, const double *x, double h, double *J, int hilos) {
    const int m = s->num_ecs, n = s->num_vars;
    #pragma omp parallel num_threads(hilos)
    {
        double *tmp = malloc((size_t)n * sizeof(double));
        if (!tmp) abort();
        memcpy(tmp, x, (size_t)n * sizeof(double));

        /* collapse(2): el bucle de ecuaciones y el de variables forman un solo
         * espacio de m*n iteraciones, así hay trabajo para todos los hilos
         * aunque haya pocas ecuaciones. */
        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                const double original = tmp[j];
                tmp[j] = original + h;
                const double f_mas = polinomio_evaluar(&s->ecs[i], tmp);
                tmp[j] = original - h;
                const double f_menos = polinomio_evaluar(&s->ecs[i], tmp);
                tmp[j] = original;
                J[(size_t)i * n + j] = (f_mas - f_menos) / (2.0 * h);
            }
        }
        free(tmp);
    }
}

void jacobiana_por_terminos_omp(const Sistema *s, const double *x, double h, double *J,
                                int hilos) {
    #pragma omp parallel for num_threads(hilos) schedule(dynamic)
    for (int i = 0; i < s->num_ecs; i++) {
        grad_por_terminos_sec(&s->ecs[i], x, h, J + (size_t)i * s->num_vars);
    }
}

int resolver_lineal(int n, double *A, double *b, double *x, int hilos) {
    for (int c = 0; c < n; c++) {
        int piv = c;
        for (int r = c + 1; r < n; r++) {
            if (fabs(A[(size_t)r * n + c]) > fabs(A[(size_t)piv * n + c])) piv = r;
        }
        if (fabs(A[(size_t)piv * n + c]) < 1e-14) return -1;
        if (piv != c) {
            for (int k = 0; k < n; k++) {
                double t = A[(size_t)c * n + k];
                A[(size_t)c * n + k] = A[(size_t)piv * n + k];
                A[(size_t)piv * n + k] = t;
            }
            double t = b[c];
            b[c] = b[piv];
            b[piv] = t;
        }
        /* Las filas por debajo del pivote se actualizan de forma independiente.
         * Solo merece la pena abrir hilos si queda trabajo suficiente. */
        #pragma omp parallel for num_threads(hilos) schedule(static) if ((n - c) > 64)
        for (int r = c + 1; r < n; r++) {
            const double f = A[(size_t)r * n + c] / A[(size_t)c * n + c];
            if (f == 0.0) continue;
            for (int k = c; k < n; k++) A[(size_t)r * n + k] -= f * A[(size_t)c * n + k];
            b[r] -= f * b[c];
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        double suma = b[i];
        for (int k = i + 1; k < n; k++) suma -= A[(size_t)i * n + k] * x[k];
        x[i] = suma / A[(size_t)i * n + i];
    }
    return 0;
}

static double norma_inf(const double *v, int n) {
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        if (fabs(v[i]) > m) m = fabs(v[i]);
    }
    return m;
}

ResultadoNewton newton_resolver(const Sistema *s, double *x, int max_iter, double tol,
                                double h, int hilos) {
    ResultadoNewton res = {0, INFINITY, 0};
    const int n = s->num_vars;
    if (s->num_ecs != n) return res;

    double *F = malloc((size_t)n * sizeof(double));
    double *J = malloc((size_t)n * n * sizeof(double));
    double *dx = malloc((size_t)n * sizeof(double));
    double *prueba = malloc((size_t)n * sizeof(double));
    if (!F || !J || !dx || !prueba) abort();

    sistema_residuo(s, x, F);
    res.residuo = norma_inf(F, n);

    while (res.iteraciones < max_iter && res.residuo > tol) {
        jacobiana_por_terminos_omp(s, x, h, J, hilos);
        for (int i = 0; i < n; i++) F[i] = -F[i];
        if (resolver_lineal(n, J, F, dx, hilos) != 0) break;

        double paso = 1.0, nuevo = INFINITY;
        for (int intento = 0; intento < 20; intento++) {
            for (int i = 0; i < n; i++) prueba[i] = x[i] + paso * dx[i];
            sistema_residuo(s, prueba, F);
            nuevo = norma_inf(F, n);
            if (nuevo < res.residuo) break;
            paso *= 0.5;
        }
        memcpy(x, prueba, (size_t)n * sizeof(double));
        res.residuo = nuevo;
        res.iteraciones++;
    }
    res.convergio = res.residuo <= tol;

    free(F);
    free(J);
    free(dx);
    free(prueba);
    return res;
}

int sistema_ejemplo_enunciado(Sistema *s) {
    enum { X = 0, Y = 1, Z = 2 };
    if (sistema_crear(s, 3, 3) != 0) return -1;

    /* 3x^2 + yz + z + 5 = 2 */
    Polinomio *e = &s->ecs[0];
    if (polinomio_crear(e, 3, 4) != 0) return -1;
    polinomio_agregar_termino(e, 0, 3.0, 1, (int[]){X}, (int[]){2});
    polinomio_agregar_termino(e, 1, 1.0, 2, (int[]){Y, Z}, (int[]){1, 1});
    polinomio_agregar_termino(e, 2, 1.0, 1, (int[]){Z}, (int[]){1});
    polinomio_agregar_termino(e, 3, 5.0, 0, NULL, NULL);
    s->rhs[0] = 2.0;

    /* 2x + yz^3 + 2x + 2 = 0 (se deja tal cual, con los dos términos en x) */
    e = &s->ecs[1];
    if (polinomio_crear(e, 3, 4) != 0) return -1;
    polinomio_agregar_termino(e, 0, 2.0, 1, (int[]){X}, (int[]){1});
    polinomio_agregar_termino(e, 1, 1.0, 2, (int[]){Y, Z}, (int[]){1, 3});
    polinomio_agregar_termino(e, 2, 2.0, 1, (int[]){X}, (int[]){1});
    polinomio_agregar_termino(e, 3, 2.0, 0, NULL, NULL);
    s->rhs[1] = 0.0;

    /* 3xz + 8yz + z^2 + 1 = 4 */
    e = &s->ecs[2];
    if (polinomio_crear(e, 3, 4) != 0) return -1;
    polinomio_agregar_termino(e, 0, 3.0, 2, (int[]){X, Z}, (int[]){1, 1});
    polinomio_agregar_termino(e, 1, 8.0, 2, (int[]){Y, Z}, (int[]){1, 1});
    polinomio_agregar_termino(e, 2, 1.0, 1, (int[]){Z}, (int[]){2});
    polinomio_agregar_termino(e, 3, 1.0, 0, NULL, NULL);
    s->rhs[2] = 4.0;
    return 0;
}

int sistema_aleatorio(Sistema *s, int n, int terms_por_ec, Rng *rng, double *x_sol) {
    if (terms_por_ec < 1 || sistema_crear(s, n, n) != 0) return -1;
    punto_aleatorio(x_sol, n, 1.0, rng);

    for (int i = 0; i < n; i++) {
        Polinomio *e = &s->ecs[i];
        if (polinomio_aleatorio(e, n, terms_por_ec + 1, 3, 2, 1.0, rng) != 0) return -1;
        /* Cada término aleatorio aporta como mucho 2 a cada derivada en [-1, 1]
         * (coeficiente <= 1, exponente <= 2) y tiene hasta 3 variables, así que
         * con un coeficiente mayor que 6 * terms_por_ec la diagonal domina. */
        const double diagonal = 6.0 * terms_por_ec + 10.0 + rng_uniforme(rng, 1.0);
        polinomio_agregar_termino(e, terms_por_ec, diagonal, 1, (int[]){i}, (int[]){1});
        s->rhs[i] = polinomio_evaluar(e, x_sol);
    }
    return 0;
}
