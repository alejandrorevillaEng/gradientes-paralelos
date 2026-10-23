#include "polinomio.h"

#include <stdlib.h>
#include <string.h>

void rng_iniciar(Rng *rng, uint64_t semilla) {
    rng->estado = semilla;
}

uint64_t rng_siguiente(Rng *rng) {
    uint64_t z = (rng->estado += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

double rng_uniforme(Rng *rng, double limite) {
    /* 53 bits de mantisa -> [0, 1) */
    double u = (double)(rng_siguiente(rng) >> 11) * (1.0 / 9007199254740992.0);
    return (2.0 * u - 1.0) * limite;
}

int rng_entero(Rng *rng, int min, int max) {
    uint64_t rango = (uint64_t)(max - min) + 1;
    return min + (int)(rng_siguiente(rng) % rango);
}

int polinomio_crear(Polinomio *p, int num_vars, int num_terms) {
    memset(p, 0, sizeof(*p));
    if (num_vars <= 0 || num_terms <= 0) return -1;
    p->num_vars = num_vars;
    p->num_terms = num_terms;
    p->coef = calloc((size_t)num_terms, sizeof(double));
    p->num_factores = calloc((size_t)num_terms, sizeof(int));
    p->var = calloc((size_t)num_terms * MAX_FACTORES, sizeof(int));
    p->exp = calloc((size_t)num_terms * MAX_FACTORES, sizeof(int));
    if (!p->coef || !p->num_factores || !p->var || !p->exp) {
        polinomio_liberar(p);
        return -1;
    }
    return 0;
}

void polinomio_liberar(Polinomio *p) {
    free(p->coef);
    free(p->num_factores);
    free(p->var);
    free(p->exp);
    memset(p, 0, sizeof(*p));
}

int polinomio_agregar_termino(Polinomio *p, int t, double coef, int num_factores,
                              const int *vars, const int *exps) {
    if (t < 0 || t >= p->num_terms || num_factores < 0 || num_factores > MAX_FACTORES) {
        return -1;
    }
    /* Un índice de variable fuera de rango haría leer fuera de x al evaluar. */
    for (int k = 0; k < num_factores; k++) {
        if (vars[k] < 0 || vars[k] >= p->num_vars || exps[k] < 1) return -1;
    }
    p->coef[t] = coef;
    p->num_factores[t] = num_factores;
    for (int k = 0; k < MAX_FACTORES; k++) {
        p->var[t * MAX_FACTORES + k] = k < num_factores ? vars[k] : 0;
        p->exp[t * MAX_FACTORES + k] = k < num_factores ? exps[k] : 0;
    }
    return t;
}

static int contiene(const int *v, int n, int x) {
    for (int i = 0; i < n; i++) {
        if (v[i] == x) return 1;
    }
    return 0;
}

int polinomio_aleatorio(Polinomio *p, int num_vars, int num_terms,
                        int max_factores, int max_exp, double rango_coef, Rng *rng) {
    if (max_factores < 1 || max_factores > MAX_FACTORES || max_exp < 1) return -1;
    if (max_factores > num_vars) max_factores = num_vars;
    if (polinomio_crear(p, num_vars, num_terms) != 0) return -1;

    int vars[MAX_FACTORES], exps[MAX_FACTORES];
    for (int t = 0; t < num_terms; t++) {
        double coef = rng_uniforme(rng, rango_coef);
        int nf = rng_entero(rng, 1, max_factores);
        for (int k = 0; k < nf; k++) {
            int v;
            do {
                v = rng_entero(rng, 0, num_vars - 1);
            } while (contiene(vars, k, v));
            vars[k] = v;
            exps[k] = rng_entero(rng, 1, max_exp);
        }
        polinomio_agregar_termino(p, t, coef, nf, vars, exps);
    }
    return 0;
}

void punto_aleatorio(double *x, int n, double rango, Rng *rng) {
    for (int i = 0; i < n; i++) x[i] = rng_uniforme(rng, rango);
}

static double potencia(double base, int e) {
    double r = 1.0;
    for (int i = 0; i < e; i++) r *= base;
    return r;
}

double polinomio_evaluar_termino(const Polinomio *p, int t, const double *x) {
    double valor = p->coef[t];
    const int base = t * MAX_FACTORES;
    for (int k = 0; k < p->num_factores[t]; k++) {
        valor *= potencia(x[p->var[base + k]], p->exp[base + k]);
    }
    return valor;
}

double polinomio_evaluar(const Polinomio *p, const double *x) {
    double suma = 0.0;
    for (int t = 0; t < p->num_terms; t++) {
        suma += polinomio_evaluar_termino(p, t, x);
    }
    return suma;
}

void polinomio_gradiente_exacto(const Polinomio *p, const double *x, double *grad) {
    memset(grad, 0, (size_t)p->num_vars * sizeof(double));
    for (int t = 0; t < p->num_terms; t++) {
        const int base = t * MAX_FACTORES;
        const int nf = p->num_factores[t];
        for (int k = 0; k < nf; k++) {
            int e = p->exp[base + k];
            /* d/dx_v (c * x_v^e * resto) = c * e * x_v^(e-1) * resto */
            double d = p->coef[t] * e * potencia(x[p->var[base + k]], e - 1);
            for (int j = 0; j < nf; j++) {
                if (j != k) d *= potencia(x[p->var[base + j]], p->exp[base + j]);
            }
            grad[p->var[base + k]] += d;
        }
    }
}

double polinomio_como_funcion(const double *x, const void *ctx) {
    return polinomio_evaluar((const Polinomio *)ctx, x);
}
