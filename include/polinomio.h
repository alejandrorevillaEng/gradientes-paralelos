#ifndef POLINOMIO_H
#define POLINOMIO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Número máximo de factores (variable^exponente) por término. */
#define MAX_FACTORES 4

/*
 * Polinomio multivariable guardado en arrays planos (estilo CSR).
 * Así se copia a la GPU con un cudaMemcpy por array y los hilos leen
 * posiciones contiguas.
 *
 * El término t vale coef[t] * prod_k x[var[t*MAX_FACTORES+k]]^exp[t*MAX_FACTORES+k]
 * para k < num_factores[t]. Un término sin factores es una constante.
 */
typedef struct {
    int num_vars;
    int num_terms;
    double *coef;
    int *num_factores;
    int *var;
    int *exp;
} Polinomio;

/* Generador pseudoaleatorio propio (splitmix64): mismos datos en Linux,
 * Windows y en la GPU, cosa que rand() no garantiza. */
typedef struct {
    uint64_t estado;
} Rng;

void rng_iniciar(Rng *rng, uint64_t semilla);
uint64_t rng_siguiente(Rng *rng);
double rng_uniforme(Rng *rng, double limite);  /* en [-limite, limite) */
int rng_entero(Rng *rng, int min, int max);      /* en [min, max] */

/* Reserva y libera. Devuelven 0 si todo va bien. */
int polinomio_crear(Polinomio *p, int num_vars, int num_terms);
void polinomio_liberar(Polinomio *p);

/* Añade un término. Devuelve el índice del término o -1 si no cabe. */
int polinomio_agregar_termino(Polinomio *p, int t, double coef, int num_factores,
                              const int *vars, const int *exps);

/*
 * Polinomio aleatorio como el de la práctica original: cada término tiene
 * entre 1 y max_factores variables distintas con exponente entre 1 y max_exp.
 */
int polinomio_aleatorio(Polinomio *p, int num_vars, int num_terms,
                        int max_factores, int max_exp, double rango_coef, Rng *rng);

void punto_aleatorio(double *x, int n, double rango, Rng *rng);

double polinomio_evaluar(const Polinomio *p, const double *x);
double polinomio_evaluar_termino(const Polinomio *p, int t, const double *x);

/* Gradiente analítico (derivando término a término). Sirve para validar. */
void polinomio_gradiente_exacto(const Polinomio *p, const double *x, double *grad);

/* Firma genérica de función escalar f: R^n -> R, para el gradiente "caja negra". */
typedef double (*FuncionEscalar)(const double *x, const void *ctx);

/* Adaptador para usar un Polinomio como FuncionEscalar. */
double polinomio_como_funcion(const double *x, const void *ctx);

#ifdef __cplusplus
}
#endif

#endif
