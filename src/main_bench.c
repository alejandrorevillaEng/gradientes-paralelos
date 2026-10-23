/*
 * gradbench: banco de pruebas de las versiones secuencial y OpenMP.
 *
 *   gradbench --experimento todo --csv resultados/cpu.csv
 *
 * Todas las versiones se validan contra el gradiente analítico y se informa
 * del error relativo junto al tiempo.
 */
#include "cronometro.h"
#include "gradiente.h"
#include "polinomio.h"
#include "sistema.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_REPS 64
#define MAX_LISTA 32

typedef struct {
    int vars;
    int terms;
    int max_exp;
    int reps;
    unsigned long long semilla;
    int hilos[MAX_LISTA];
    int num_hilos;
    int tamanos[MAX_LISTA];
    int num_tamanos;
    const char *experimento;
    const char *csv;
} Opciones;

static FILE *salida_csv = NULL;

static int comparar_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double mediana(double *v, int n) {
    qsort(v, (size_t)n, sizeof(double), comparar_double);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* Duración mínima de una medida. Si la instrucción tarda menos, se repite en
 * bucle y se divide, para no medir solo el ruido del reloj. */
#define DURACION_MINIMA 0.05

/* Ejecuta la instrucción `reps` veces y guarda la mediana del tiempo. */
#define MEDIR(reps, destino, instruccion)                                   \
    do {                                                                    \
        double t0_ = cronometro_ahora();                                    \
        instruccion;                                                        \
        const double una_ = cronometro_ahora() - t0_;                       \
        const int veces_ = una_ >= DURACION_MINIMA                          \
                               ? 1                                          \
                               : (int)(DURACION_MINIMA / (una_ + 1e-9)) + 1; \
        double tiempos_[MAX_REPS];                                          \
        for (int r_ = 0; r_ < (reps); r_++) {                               \
            t0_ = cronometro_ahora();                                       \
            for (int v_ = 0; v_ < veces_; v_++) {                           \
                instruccion;                                                \
            }                                                               \
            tiempos_[r_] = (cronometro_ahora() - t0_) / veces_;             \
        }                                                                   \
        (destino) = mediana(tiempos_, (reps));                              \
    } while (0)

/* Entero positivo; -1 si el texto no es un número válido. */
static int leer_entero(const char *texto) {
    char *fin;
    long v = strtol(texto, &fin, 10);
    return (fin == texto || *fin != '\0' || v <= 0 || v > 100000000L) ? -1 : (int)v;
}

/* Lista separada por comas, p. ej. "1,2,4,8". Ignora los valores no válidos. */
static int leer_lista(const char *texto, int *lista) {
    int n = 0;
    const char *p = texto;
    while (*p && n < MAX_LISTA) {
        char *fin;
        long v = strtol(p, &fin, 10);
        if (fin == p) break;
        if (v > 0 && v <= 100000000L) lista[n++] = (int)v;
        p = *fin == ',' ? fin + 1 : fin;
        if (*fin != ',' && *fin != '\0') break;
    }
    return n;
}

static void ayuda(void) {
    puts("Uso: gradbench [opciones]\n"
         "  --experimento E  escalado | tamano | anidado | metodos | sistemas | todo (todo)\n"
         "  --vars N         numero de variables (500)\n"
         "  --terms T        numero de terminos (20000)\n"
         "  --max-exp E      exponente maximo de cada variable (2)\n"
         "  --hilos L        lista de hilos, p. ej. 1,2,4,8 (potencias de 2 hasta los nucleos)\n"
         "  --tamanos L      terminos para el experimento de tamano (20000,40000,80000,160000)\n"
         "  --reps R         repeticiones por medida; se guarda la mediana (3)\n"
         "  --semilla S      semilla del generador (2025)\n"
         "  --csv FICHERO    guarda los resultados en CSV");
}

static int leer_opciones(int argc, char **argv, Opciones *o) {
    *o = (Opciones){500, 20000, 2, 3, 2025ULL, {0}, 0, {20000, 40000, 80000, 160000}, 4,
                    "todo", NULL};
    /* Las opciones van por parejas: --nombre valor. */
    for (int i = 1; i < argc; i += 2) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--help") || !strcmp(a, "-h")) {
            ayuda();
            exit(0);
        }
        if (!v) {
            fprintf(stderr, "Falta el valor de %s\n", a);
            return -1;
        }
        if (!strcmp(a, "--experimento")) o->experimento = v;
        else if (!strcmp(a, "--vars")) o->vars = leer_entero(v);
        else if (!strcmp(a, "--terms")) o->terms = leer_entero(v);
        else if (!strcmp(a, "--max-exp")) o->max_exp = leer_entero(v);
        else if (!strcmp(a, "--reps")) o->reps = leer_entero(v);
        else if (!strcmp(a, "--semilla")) o->semilla = strtoull(v, NULL, 10);
        else if (!strcmp(a, "--hilos")) o->num_hilos = leer_lista(v, o->hilos);
        else if (!strcmp(a, "--tamanos")) o->num_tamanos = leer_lista(v, o->tamanos);
        else if (!strcmp(a, "--csv")) o->csv = v;
        else {
            fprintf(stderr, "Opcion desconocida: %s\n", a);
            return -1;
        }
    }
    if (o->vars < 1 || o->terms < 1 || o->max_exp < 1 || o->reps < 1 || o->reps > MAX_REPS) {
        fprintf(stderr, "Valores fuera de rango\n");
        return -1;
    }
    if (o->num_hilos == 0) {
        const int nucleos = omp_get_num_procs();
        for (int h = 1; h <= nucleos && o->num_hilos < MAX_LISTA; h *= 2) o->hilos[o->num_hilos++] = h;
        if (o->hilos[o->num_hilos - 1] != nucleos) o->hilos[o->num_hilos++] = nucleos;
    }
    return 0;
}

static void registrar(const char *exp, const char *metodo, int vars, int terms, int h_ext,
                      int h_int, double t, double t_ref, double err) {
    const double speedup = t_ref / t;
    const double eficiencia = speedup / (h_ext * h_int);
    printf("  %-26s %3dx%-3d %11.3f ms  speedup %8.2fx  efic. %5.1f %%  err %.1e\n", metodo,
           h_ext, h_int, 1e3 * t, speedup, 100.0 * eficiencia, err);
    if (salida_csv) {
        fprintf(salida_csv, "%s,%s,%d,%d,%d,%d,%.8f,%.4f,%.4f,%.3e\n", exp, metodo, vars, terms,
                h_ext, h_int, t, speedup, eficiencia, err);
        fflush(salida_csv);
    }
}

typedef struct {
    Polinomio p;
    double *x;
    double *exacto;
    double *grad;
} Caso;

static int crear_caso(Caso *c, const Opciones *o, int terms) {
    Rng rng;
    rng_iniciar(&rng, o->semilla);
    if (polinomio_aleatorio(&c->p, o->vars, terms, 3, o->max_exp, 10.0, &rng) != 0) return -1;
    c->x = malloc((size_t)o->vars * sizeof(double));
    c->exacto = malloc((size_t)o->vars * sizeof(double));
    c->grad = malloc((size_t)o->vars * sizeof(double));
    if (!c->x || !c->exacto || !c->grad) return -1;
    punto_aleatorio(c->x, o->vars, 2.0, &rng);
    polinomio_gradiente_exacto(&c->p, c->x, c->exacto);
    return 0;
}

static void liberar_caso(Caso *c) {
    polinomio_liberar(&c->p);
    free(c->x);
    free(c->exacto);
    free(c->grad);
}

static double tiempo_secuencial(const Opciones *o, Caso *c, double *err) {
    double t;
    MEDIR(o->reps, t, grad_directo_sec(&c->p, c->x, PASO_H_DEFECTO, c->grad));
    *err = error_relativo(c->grad, c->exacto, c->p.num_vars);
    return t;
}

static void exp_escalado(const Opciones *o) {
    printf("\n== Escalabilidad con el numero de hilos (%d vars, %d terminos) ==\n", o->vars, o->terms);
    Caso c;
    if (crear_caso(&c, o, o->terms) != 0) exit(1);
    double err, t_ref = tiempo_secuencial(o, &c, &err);
    registrar("escalado", "secuencial", o->vars, o->terms, 1, 1, t_ref, t_ref, err);
    for (int k = 0; k < o->num_hilos; k++) {
        const int h = o->hilos[k];
        double t;
        MEDIR(o->reps, t, grad_directo_omp(&c.p, c.x, PASO_H_DEFECTO, c.grad, h));
        registrar("escalado", "omp_nivel1", o->vars, o->terms, h, 1, t, t_ref,
                  error_relativo(c.grad, c.exacto, o->vars));
    }
    liberar_caso(&c);
}

static void exp_tamano(const Opciones *o) {
    const int h = o->hilos[o->num_hilos - 1];
    printf("\n== Tamano del problema (%d vars, %d hilos) ==\n", o->vars, h);
    for (int k = 0; k < o->num_tamanos; k++) {
        Caso c;
        if (crear_caso(&c, o, o->tamanos[k]) != 0) exit(1);
        double err, t_ref = tiempo_secuencial(o, &c, &err);
        registrar("tamano", "secuencial", o->vars, o->tamanos[k], 1, 1, t_ref, t_ref, err);
        double t;
        MEDIR(o->reps, t, grad_directo_omp(&c.p, c.x, PASO_H_DEFECTO, c.grad, h));
        registrar("tamano", "omp_nivel1", o->vars, o->tamanos[k], h, 1, t, t_ref,
                  error_relativo(c.grad, c.exacto, o->vars));
        liberar_caso(&c);
    }
}

static void exp_anidado(const Opciones *o) {
    const int total = o->hilos[o->num_hilos - 1];
    printf("\n== Paralelismo anidado: reparto externo x interno con %d hilos ==\n", total);
    Caso c;
    if (crear_caso(&c, o, o->terms) != 0) exit(1);
    double err, t_ref = tiempo_secuencial(o, &c, &err);
    registrar("anidado", "secuencial", o->vars, o->terms, 1, 1, t_ref, t_ref, err);
    for (int ext = 1; ext <= total; ext++) {
        if (total % ext) continue;
        const int in = total / ext;
        double t;
        MEDIR(o->reps, t, grad_directo_anidado(&c.p, c.x, PASO_H_DEFECTO, c.grad, ext, in));
        registrar("anidado", "omp_anidado", o->vars, o->terms, ext, in, t, t_ref,
                  error_relativo(c.grad, c.exacto, o->vars));
    }
    /* El caso de la practica original: el mismo numero de hilos en los dos
     * niveles, que en realidad lanza total*total hilos. */
    if (total > 1) {
        double t;
        MEDIR(o->reps, t, grad_directo_anidado(&c.p, c.x, PASO_H_DEFECTO, c.grad, total, total));
        registrar("anidado", "omp_anidado_sobresuscrito", o->vars, o->terms, total, total, t,
                  t_ref, error_relativo(c.grad, c.exacto, o->vars));
    }
    liberar_caso(&c);
}

static void exp_metodos(const Opciones *o) {
    const int h = o->hilos[o->num_hilos - 1];
    printf("\n== Metodo directo frente a metodo por terminos (%d hilos) ==\n", h);
    Caso c;
    if (crear_caso(&c, o, o->terms) != 0) exit(1);
    double err, t_ref = tiempo_secuencial(o, &c, &err), t;
    registrar("metodos", "directo_sec", o->vars, o->terms, 1, 1, t_ref, t_ref, err);
    MEDIR(o->reps, t, grad_directo_omp(&c.p, c.x, PASO_H_DEFECTO, c.grad, h));
    registrar("metodos", "directo_omp", o->vars, o->terms, h, 1, t, t_ref,
              error_relativo(c.grad, c.exacto, o->vars));
    MEDIR(o->reps, t, grad_por_terminos_sec(&c.p, c.x, PASO_H_DEFECTO, c.grad));
    registrar("metodos", "por_terminos_sec", o->vars, o->terms, 1, 1, t, t_ref,
              error_relativo(c.grad, c.exacto, o->vars));
    MEDIR(o->reps, t, grad_por_terminos_omp(&c.p, c.x, PASO_H_DEFECTO, c.grad, h));
    registrar("metodos", "por_terminos_omp", o->vars, o->terms, h, 1, t, t_ref,
              error_relativo(c.grad, c.exacto, o->vars));
    liberar_caso(&c);
}

static void exp_sistemas(const Opciones *o) {
    const int h = o->hilos[o->num_hilos - 1];
    const int n = 200, terms = 100;
    printf("\n== Sistemas de ecuaciones: jacobiana y Newton (%d ecuaciones, %d terminos/ec.) ==\n",
           n, terms);
    Rng rng;
    rng_iniciar(&rng, o->semilla);
    Sistema s;
    double *sol = malloc((size_t)n * sizeof(double));
    double *J = malloc((size_t)n * n * sizeof(double));
    double *Jex = malloc((size_t)n * n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    if (!sol || !J || !Jex || !x || sistema_aleatorio(&s, n, terms, &rng, sol) != 0) exit(1);
    for (int i = 0; i < n; i++) x[i] = sol[i] + rng_uniforme(&rng, 0.3);
    jacobiana_exacta(&s, x, Jex);

    double t_ref, t;
    MEDIR(o->reps, t_ref, jacobiana_directa_sec(&s, x, PASO_H_DEFECTO, J));
    registrar("sistemas", "jacobiana_directa_sec", n, terms, 1, 1, t_ref, t_ref,
              error_relativo(J, Jex, n * n));
    MEDIR(o->reps, t, jacobiana_directa_omp(&s, x, PASO_H_DEFECTO, J, h));
    registrar("sistemas", "jacobiana_directa_omp", n, terms, h, 1, t, t_ref,
              error_relativo(J, Jex, n * n));
    MEDIR(o->reps, t, jacobiana_por_terminos_omp(&s, x, PASO_H_DEFECTO, J, h));
    registrar("sistemas", "jacobiana_terminos_omp", n, terms, h, 1, t, t_ref,
              error_relativo(J, Jex, n * n));

    double t0 = cronometro_ahora();
    ResultadoNewton r = newton_resolver(&s, x, 50, 1e-10, PASO_H_DEFECTO, h);
    printf("  Newton: %s en %d iteraciones, residuo %.2e, %.4f s, error frente a la solucion %.1e\n",
           r.convergio ? "converge" : "NO converge", r.iteraciones, r.residuo,
           cronometro_ahora() - t0, error_relativo(x, sol, n));

    sistema_liberar(&s);
    free(sol);
    free(J);
    free(Jex);
    free(x);
}

int main(int argc, char **argv) {
    Opciones o;
    if (leer_opciones(argc, argv, &o) != 0) {
        ayuda();
        return 2;
    }
    if (o.csv) {
        salida_csv = fopen(o.csv, "w");
        if (!salida_csv) {
            perror(o.csv);
            return 1;
        }
        fprintf(salida_csv,
                "experimento,metodo,vars,terminos,hilos_ext,hilos_int,tiempo_s,speedup,eficiencia,error_rel\n");
    }
    printf("gradbench | %d nucleos logicos | reps=%d (mediana) | semilla=%llu\n", omp_get_num_procs(),
           o.reps, o.semilla);

    const char *e = o.experimento;
    const int todo = !strcmp(e, "todo");
    if (todo || !strcmp(e, "escalado")) exp_escalado(&o);
    if (todo || !strcmp(e, "tamano")) exp_tamano(&o);
    if (todo || !strcmp(e, "anidado")) exp_anidado(&o);
    if (todo || !strcmp(e, "metodos")) exp_metodos(&o);
    if (todo || !strcmp(e, "sistemas")) exp_sistemas(&o);

    if (salida_csv) fclose(salida_csv);
    return 0;
}
