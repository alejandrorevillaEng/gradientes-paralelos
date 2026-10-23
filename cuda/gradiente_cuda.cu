/*
 * gradbench_cuda: gradiente de un polinomio multivariable en GPU.
 *
 * Compara cuatro kernels con la versión secuencial y la mejor OpenMP:
 *   directo_original  un hilo por variable que acumula f(x+h) y f(x-h) sobre
 *                     todos los términos (el kernel de la práctica entregada)
 *   directo           igual, pero solo suma la diferencia de los términos que
 *                     contienen la variable: misma salida, menos operaciones y
 *                     sin restar dos números enormes
 *   directo_tiled     el anterior cargando los términos por bloques en memoria
 *                     compartida (tiling)
 *   por_terminos      un hilo por término que suma su aportación a cada una de
 *                     sus variables con atomicAdd (requiere sm_60 o superior)
 *
 * Tiempos con cudaEvent: copia H->D, kernel y copia D->H por separado.
 *
 * El experimento "precision" repite el kernel directo en FP64 y en FP32. Las
 * GeForce tienen una fracción pequeña de unidades de doble precisión, pero con
 * diferencias finitas FP32 obliga a usar un paso h mucho mayor.
 */
#include "gradiente.h"
#include "polinomio.h"
#include "sistema.h"

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_OK(llamada)                                                              \
    do {                                                                              \
        cudaError_t e_ = (llamada);                                                   \
        if (e_ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

struct PolinomioDev {
    int num_vars;
    int num_terms;
    const double *coef;
    const int *num_factores;
    const int *var;
    const int *exp;
};

__device__ __forceinline__ double potencia(double b, int e) {
    double r = 1.0;
    for (int i = 0; i < e; i++) r *= b;
    return r;
}

__global__ void k_directo_original(PolinomioDev p, const double *x, double h, double *grad) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.num_vars) return;
    double f_mas = 0.0, f_menos = 0.0;
    for (int t = 0; t < p.num_terms; t++) {
        double vm = p.coef[t], vn = p.coef[t];
        for (int k = 0; k < p.num_factores[t]; k++) {
            const int v = p.var[t * MAX_FACTORES + k];
            const int e = p.exp[t * MAX_FACTORES + k];
            if (v == i) {
                vm *= potencia(x[v] + h, e);
                vn *= potencia(x[v] - h, e);
            } else {
                const double q = potencia(x[v], e);
                vm *= q;
                vn *= q;
            }
        }
        f_mas += vm;
        f_menos += vn;
    }
    grad[i] = (f_mas - f_menos) / (2.0 * h);
}

/* Diferencia centrada del término (coef, vars, exps) respecto a la variable i,
 * o 0 si el término no contiene i. */
__device__ __forceinline__ double diferencia_termino(double coef, int nf, const int *vars,
                                                     const int *exps, const double *x, int i,
                                                     double h) {
    int pos = -1;
    for (int k = 0; k < nf; k++) {
        if (vars[k] == i) pos = k;
    }
    if (pos < 0) return 0.0;
    double resto = coef;
    for (int k = 0; k < nf; k++) {
        if (k != pos) resto *= potencia(x[vars[k]], exps[k]);
    }
    const double xi = x[i];
    return resto * (potencia(xi + h, exps[pos]) - potencia(xi - h, exps[pos])) / (2.0 * h);
}

__global__ void k_directo(PolinomioDev p, const double *x, double h, double *grad) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.num_vars) return;
    double suma = 0.0;
    for (int t = 0; t < p.num_terms; t++) {
        suma += diferencia_termino(p.coef[t], p.num_factores[t], p.var + t * MAX_FACTORES,
                                   p.exp + t * MAX_FACTORES, x, i, h);
    }
    grad[i] = suma;
}

/* Cada bloque carga blockDim.x términos en memoria compartida y todos sus
 * hilos los recorren antes de cargar el siguiente trozo. */
__global__ void k_directo_tiled(PolinomioDev p, const double *x, double h, double *grad) {
    extern __shared__ __align__(8) unsigned char memoria[];
    const int tile = blockDim.x;
    double *s_coef = reinterpret_cast<double *>(memoria);
    int *s_nf = reinterpret_cast<int *>(s_coef + tile);
    int *s_var = s_nf + tile;
    int *s_exp = s_var + tile * MAX_FACTORES;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double suma = 0.0;
    for (int base = 0; base < p.num_terms; base += tile) {
        const int t = base + threadIdx.x;
        if (t < p.num_terms) {
            s_coef[threadIdx.x] = p.coef[t];
            s_nf[threadIdx.x] = p.num_factores[t];
            for (int k = 0; k < MAX_FACTORES; k++) {
                s_var[threadIdx.x * MAX_FACTORES + k] = p.var[t * MAX_FACTORES + k];
                s_exp[threadIdx.x * MAX_FACTORES + k] = p.exp[t * MAX_FACTORES + k];
            }
        }
        __syncthreads();
        const int n = min(tile, p.num_terms - base);
        if (i < p.num_vars) {
            for (int j = 0; j < n; j++) {
                suma += diferencia_termino(s_coef[j], s_nf[j], s_var + j * MAX_FACTORES,
                                           s_exp + j * MAX_FACTORES, x, i, h);
            }
        }
        __syncthreads();
    }
    if (i < p.num_vars) grad[i] = suma;
}

/* Un hilo por término. fila[t] indica en qué fila de la salida se acumula
 * (todo ceros para un gradiente; la ecuación para una jacobiana). */
__global__ void k_por_terminos(PolinomioDev p, const int *fila, const double *x, double h,
                               double *salida) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= p.num_terms) return;
    const int nf = p.num_factores[t];
    const int *vars = p.var + t * MAX_FACTORES;
    const int *exps = p.exp + t * MAX_FACTORES;
    const size_t desplazamiento = fila ? (size_t)fila[t] * p.num_vars : 0;
    for (int k = 0; k < nf; k++) {
        const double d = diferencia_termino(p.coef[t], nf, vars, exps, x, vars[k], h);
        atomicAdd(&salida[desplazamiento + vars[k]], d);
    }
}

/* El kernel directo para cualquier tipo de coma flotante (experimento de precisión). */
template <typename T>
__device__ __forceinline__ T potencia_t(T b, int e) {
    T r = 1;
    for (int i = 0; i < e; i++) r *= b;
    return r;
}

template <typename T>
__global__ void k_directo_t(int num_vars, int num_terms, const T *coef, const int *num_factores,
                            const int *var, const int *exp, const T *x, T h, T *grad) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vars) return;
    T suma = 0;
    for (int t = 0; t < num_terms; t++) {
        const int nf = num_factores[t];
        const int *vars = var + t * MAX_FACTORES;
        const int *exps = exp + t * MAX_FACTORES;
        int pos = -1;
        for (int k = 0; k < nf; k++) {
            if (vars[k] == i) pos = k;
        }
        if (pos < 0) continue;
        T resto = coef[t];
        for (int k = 0; k < nf; k++) {
            if (k != pos) resto *= potencia_t(x[vars[k]], exps[k]);
        }
        const T xi = x[i];
        suma += resto * (potencia_t(xi + h, exps[pos]) - potencia_t(xi - h, exps[pos])) / (T(2) * h);
    }
    grad[i] = suma;
}

/* ------------------------------------------------------------------------ */

struct Opciones {
    int vars = 500;
    int terms = 20000;
    int max_exp = 2;
    int reps = 5;
    unsigned long long semilla = 2025ULL;
    std::vector<int> bloques{32, 64, 128, 256, 512, 1024};
    std::vector<int> lista_vars{500, 2000, 8000, 32000};
    const char *csv = nullptr;
    std::string experimento = "todo";
};

static FILE *salida_csv = nullptr;

static std::vector<int> leer_lista(const char *texto) {
    std::vector<int> v;
    std::string s(texto);
    size_t ini = 0;
    while (ini <= s.size()) {
        size_t fin = s.find(',', ini);
        if (fin == std::string::npos) fin = s.size();
        int n = static_cast<int>(std::strtol(s.substr(ini, fin - ini).c_str(), nullptr, 10));
        if (n > 0) v.push_back(n);
        ini = fin + 1;
    }
    return v;
}

static double mediana(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* Copia de un polinomio en la GPU. */
struct PolinomioGpu {
    PolinomioDev dev{};
    double *coef = nullptr;
    int *nf = nullptr, *var = nullptr, *exp = nullptr;

    void reservar(int vars, int terms) {
        CUDA_OK(cudaMalloc(&coef, terms * sizeof(double)));
        CUDA_OK(cudaMalloc(&nf, terms * sizeof(int)));
        CUDA_OK(cudaMalloc(&var, (size_t)terms * MAX_FACTORES * sizeof(int)));
        CUDA_OK(cudaMalloc(&exp, (size_t)terms * MAX_FACTORES * sizeof(int)));
        dev = {vars, terms, coef, nf, var, exp};
    }
    void copiar(const Polinomio &p, cudaStream_t s) {
        const int T = p.num_terms;
        CUDA_OK(cudaMemcpyAsync(coef, p.coef, T * sizeof(double), cudaMemcpyHostToDevice, s));
        CUDA_OK(cudaMemcpyAsync(nf, p.num_factores, T * sizeof(int), cudaMemcpyHostToDevice, s));
        CUDA_OK(cudaMemcpyAsync(var, p.var, (size_t)T * MAX_FACTORES * sizeof(int),
                                cudaMemcpyHostToDevice, s));
        CUDA_OK(cudaMemcpyAsync(exp, p.exp, (size_t)T * MAX_FACTORES * sizeof(int),
                                cudaMemcpyHostToDevice, s));
    }
    void liberar() {
        cudaFree(coef);
        cudaFree(nf);
        cudaFree(var);
        cudaFree(exp);
    }
};

/* Pasa los arrays del polinomio a memoria fijada (pinned) para que las copias
 * vayan por DMA directo. */
static void fijar_memoria(Polinomio &p) {
    const size_t T = p.num_terms;
    CUDA_OK(cudaHostRegister(p.coef, T * sizeof(double), cudaHostRegisterDefault));
    CUDA_OK(cudaHostRegister(p.num_factores, T * sizeof(int), cudaHostRegisterDefault));
    CUDA_OK(cudaHostRegister(p.var, T * MAX_FACTORES * sizeof(int), cudaHostRegisterDefault));
    CUDA_OK(cudaHostRegister(p.exp, T * MAX_FACTORES * sizeof(int), cudaHostRegisterDefault));
}

static void soltar_memoria(Polinomio &p) {
    cudaHostUnregister(p.coef);
    cudaHostUnregister(p.num_factores);
    cudaHostUnregister(p.var);
    cudaHostUnregister(p.exp);
}

enum Kernel { ORIGINAL, DIRECTO, TILED, POR_TERMINOS };
static const char *NOMBRES[] = {"directo_original", "directo", "directo_tiled", "por_terminos"};

static void lanzar(Kernel k, int bloque, const PolinomioGpu &g, const double *d_x, double h,
                   double *d_grad) {
    const int n = g.dev.num_vars, T = g.dev.num_terms;
    switch (k) {
        case ORIGINAL:
            k_directo_original<<<(n + bloque - 1) / bloque, bloque>>>(g.dev, d_x, h, d_grad);
            break;
        case DIRECTO:
            k_directo<<<(n + bloque - 1) / bloque, bloque>>>(g.dev, d_x, h, d_grad);
            break;
        case TILED: {
            const size_t mem = bloque * (sizeof(double) + sizeof(int) * (1 + 2 * MAX_FACTORES));
            k_directo_tiled<<<(n + bloque - 1) / bloque, bloque, mem>>>(g.dev, d_x, h, d_grad);
            break;
        }
        case POR_TERMINOS:
            CUDA_OK(cudaMemset(d_grad, 0, n * sizeof(double)));
            k_por_terminos<<<(T + bloque - 1) / bloque, bloque>>>(g.dev, nullptr, d_x, h, d_grad);
            break;
    }
    CUDA_OK(cudaGetLastError());
}

struct Referencias {
    double t_sec_ms;  /* < 0 si no se midió */
    double t_omp_ms;
};

static void registrar(const char *exp, const char *kernel, int vars, int terms, int bloque,
                      double h2d, double ker, double d2h, const Referencias &ref, double err) {
    const double total = h2d + ker + d2h;
    const double sp_sec = ref.t_sec_ms > 0 ? ref.t_sec_ms / total : 0.0;
    const double sp_omp = ref.t_omp_ms / total;
    printf("  %-17s %6d vars bloque %4d | H->D %7.3f  kernel %9.3f  D->H %6.3f  total %9.3f ms"
           " | x%-7.1f vs sec  x%-6.2f vs omp | err %.1e\n",
           kernel, vars, bloque, h2d, ker, d2h, total, sp_sec, sp_omp, err);
    if (salida_csv) {
        fprintf(salida_csv, "%s,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3e\n", exp, kernel, vars,
                terms, bloque, h2d, ker, d2h, total, sp_sec, sp_omp, err);
        fflush(salida_csv);
    }
}

/* Mide todos los kernels con todos los tamaños de bloque para un problema. */
static void medir_problema(const char *exp, const Opciones &o, int vars, bool medir_secuencial,
                           const std::vector<int> &bloques) {
    Rng rng;
    rng_iniciar(&rng, o.semilla);
    Polinomio p;
    if (polinomio_aleatorio(&p, vars, o.terms, 3, o.max_exp, 10.0, &rng) != 0) exit(1);
    std::vector<double> x(vars), exacto(vars), grad(vars);
    punto_aleatorio(x.data(), vars, 2.0, &rng);
    polinomio_gradiente_exacto(&p, x.data(), exacto.data());

    Referencias ref{-1.0, 0.0};
    const int hilos = omp_get_num_procs();
    if (medir_secuencial) {
        double t0 = omp_get_wtime();
        grad_directo_sec(&p, x.data(), PASO_H_DEFECTO, grad.data());
        ref.t_sec_ms = 1e3 * (omp_get_wtime() - t0);
    }
    {
        double t0 = omp_get_wtime();
        grad_directo_omp(&p, x.data(), PASO_H_DEFECTO, grad.data(), hilos);
        ref.t_omp_ms = 1e3 * (omp_get_wtime() - t0);
    }
    printf("\n== %s: %d variables, %d terminos | CPU secuencial %s ms, OpenMP (%d hilos) %.1f ms ==\n",
           exp, vars, o.terms,
           medir_secuencial ? std::to_string(ref.t_sec_ms).c_str() : "(no medido)", hilos,
           ref.t_omp_ms);

    fijar_memoria(p);
    double *h_x, *h_grad;
    CUDA_OK(cudaMallocHost(&h_x, vars * sizeof(double)));
    CUDA_OK(cudaMallocHost(&h_grad, vars * sizeof(double)));
    std::memcpy(h_x, x.data(), vars * sizeof(double));

    PolinomioGpu g;
    g.reservar(vars, o.terms);
    double *d_x, *d_grad;
    CUDA_OK(cudaMalloc(&d_x, vars * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_grad, vars * sizeof(double)));

    cudaEvent_t e0, e1, e2, e3;
    CUDA_OK(cudaEventCreate(&e0));
    CUDA_OK(cudaEventCreate(&e1));
    CUDA_OK(cudaEventCreate(&e2));
    CUDA_OK(cudaEventCreate(&e3));

    /* Calentamiento: la primera llamada incluye la inicialización del contexto. */
    g.copiar(p, 0);
    lanzar(DIRECTO, 128, g, d_x, PASO_H_DEFECTO, d_grad);
    CUDA_OK(cudaDeviceSynchronize());

    for (int k = ORIGINAL; k <= POR_TERMINOS; k++) {
        for (int bloque : bloques) {
            std::vector<double> th, tk, td;
            for (int r = 0; r < o.reps; r++) {
                CUDA_OK(cudaEventRecord(e0));
                g.copiar(p, 0);
                CUDA_OK(cudaMemcpyAsync(d_x, h_x, vars * sizeof(double), cudaMemcpyHostToDevice));
                CUDA_OK(cudaEventRecord(e1));
                lanzar(static_cast<Kernel>(k), bloque, g, d_x, PASO_H_DEFECTO, d_grad);
                CUDA_OK(cudaEventRecord(e2));
                CUDA_OK(cudaMemcpyAsync(h_grad, d_grad, vars * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_OK(cudaEventRecord(e3));
                CUDA_OK(cudaEventSynchronize(e3));
                float a, b, c;
                CUDA_OK(cudaEventElapsedTime(&a, e0, e1));
                CUDA_OK(cudaEventElapsedTime(&b, e1, e2));
                CUDA_OK(cudaEventElapsedTime(&c, e2, e3));
                th.push_back(a);
                tk.push_back(b);
                td.push_back(c);
            }
            registrar(exp, NOMBRES[k], vars, o.terms, bloque, mediana(th), mediana(tk), mediana(td),
                      ref, error_relativo(h_grad, exacto.data(), vars));
        }
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(e3);
    cudaFree(d_x);
    cudaFree(d_grad);
    g.liberar();
    cudaFreeHost(h_x);
    cudaFreeHost(h_grad);
    soltar_memoria(p);
    polinomio_liberar(&p);
}

/* Mide el kernel directo con datos de tipo T y paso h. Devuelve el gradiente en double. */
template <typename T>
static void medir_tipo(const char *nombre, const Polinomio &p, const std::vector<double> &x,
                       const std::vector<double> &exacto, T h, int bloque, const Opciones &o,
                       const Referencias &ref) {
    const int n = p.num_vars, T_ = p.num_terms;
    std::vector<T> coef(p.coef, p.coef + T_), xt(x.begin(), x.end()), g(n);
    T *d_coef, *d_x, *d_grad;
    int *d_nf, *d_var, *d_exp;
    CUDA_OK(cudaMalloc(&d_coef, T_ * sizeof(T)));
    CUDA_OK(cudaMalloc(&d_x, n * sizeof(T)));
    CUDA_OK(cudaMalloc(&d_grad, n * sizeof(T)));
    CUDA_OK(cudaMalloc(&d_nf, T_ * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_var, (size_t)T_ * MAX_FACTORES * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_exp, (size_t)T_ * MAX_FACTORES * sizeof(int)));

    cudaEvent_t e0, e1, e2, e3;
    CUDA_OK(cudaEventCreate(&e0));
    CUDA_OK(cudaEventCreate(&e1));
    CUDA_OK(cudaEventCreate(&e2));
    CUDA_OK(cudaEventCreate(&e3));
    std::vector<double> th, tk, td;
    for (int r = 0; r <= o.reps; r++) {  /* la vuelta 0 es de calentamiento */
        CUDA_OK(cudaEventRecord(e0));
        CUDA_OK(cudaMemcpy(d_coef, coef.data(), T_ * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_nf, p.num_factores, T_ * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_var, p.var, (size_t)T_ * MAX_FACTORES * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_exp, p.exp, (size_t)T_ * MAX_FACTORES * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_x, xt.data(), n * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_OK(cudaEventRecord(e1));
        k_directo_t<T><<<(n + bloque - 1) / bloque, bloque>>>(n, T_, d_coef, d_nf, d_var, d_exp, d_x, h,
                                                                d_grad);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaEventRecord(e2));
        CUDA_OK(cudaMemcpy(g.data(), d_grad, n * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaEventRecord(e3));
        CUDA_OK(cudaEventSynchronize(e3));
        float a, b, c;
        CUDA_OK(cudaEventElapsedTime(&a, e0, e1));
        CUDA_OK(cudaEventElapsedTime(&b, e1, e2));
        CUDA_OK(cudaEventElapsedTime(&c, e2, e3));
        if (r > 0) {
            th.push_back(a);
            tk.push_back(b);
            td.push_back(c);
        }
    }
    std::vector<double> gd(g.begin(), g.end());
    registrar("precision", nombre, n, T_, bloque, mediana(th), mediana(tk), mediana(td), ref,
              error_relativo(gd.data(), exacto.data(), n));

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(e3);
    cudaFree(d_coef);
    cudaFree(d_x);
    cudaFree(d_grad);
    cudaFree(d_nf);
    cudaFree(d_var);
    cudaFree(d_exp);
}

/*
 * FP64 frente a FP32 con el kernel directo. Con exponentes hasta 3 la diferencia
 * centrada tiene error de truncamiento O(h^2) y de redondeo O(eps/h); el paso que
 * equilibra los dos es del orden de cbrt(eps): ~6e-6 en double y ~5e-3 en float.
 */
static void medir_precision(const Opciones &o) {
    const int vars = 8000, bloque = 256;
    Rng rng;
    rng_iniciar(&rng, o.semilla);
    Polinomio p;
    if (polinomio_aleatorio(&p, vars, o.terms, 3, 3, 10.0, &rng) != 0) exit(1);
    std::vector<double> x(vars), exacto(vars), grad(vars);
    punto_aleatorio(x.data(), vars, 2.0, &rng);
    polinomio_gradiente_exacto(&p, x.data(), exacto.data());

    double t0 = omp_get_wtime();
    grad_directo_omp(&p, x.data(), PASO_H_DEFECTO, grad.data(), omp_get_num_procs());
    const Referencias ref{-1.0, 1e3 * (omp_get_wtime() - t0)};
    printf("\n== precision: %d variables, %d terminos, exponentes hasta 3 | OpenMP %.1f ms ==\n", vars,
           o.terms, ref.t_omp_ms);

    medir_tipo<double>("fp64", p, x, exacto, PASO_H_DEFECTO, bloque, o, ref);
    medir_tipo<float>("fp32_h_1e-6", p, x, exacto, 1e-6f, bloque, o, ref);
    medir_tipo<float>("fp32_h_ajustado", p, x, exacto, std::cbrt(FLT_EPSILON), bloque, o, ref);
    polinomio_liberar(&p);
}

/* Jacobiana de un sistema aleatorio n x n con el kernel por términos. */
static void medir_jacobiana(const Opciones &o) {
    const int n = 2000, terms = 100, bloque = 256;
    Rng rng;
    rng_iniciar(&rng, o.semilla);
    Sistema s;
    std::vector<double> sol(n), x(n);
    if (sistema_aleatorio(&s, n, terms, &rng, sol.data()) != 0) exit(1);
    for (int i = 0; i < n; i++) x[i] = sol[i] + rng_uniforme(&rng, 0.3);

    std::vector<double> J((size_t)n * n), Jex((size_t)n * n);
    jacobiana_exacta(&s, x.data(), Jex.data());
    double t0 = omp_get_wtime();
    jacobiana_por_terminos_omp(&s, x.data(), PASO_H_DEFECTO, J.data(), omp_get_num_procs());
    const double t_omp = 1e3 * (omp_get_wtime() - t0);

    /* Todos los términos de todas las ecuaciones en un único polinomio plano,
     * con fila[t] = ecuación a la que pertenece el término. */
    int total = 0;
    for (int i = 0; i < n; i++) total += s.ecs[i].num_terms;
    Polinomio plano;
    if (polinomio_crear(&plano, n, total) != 0) exit(1);
    std::vector<int> fila(total);
    for (int i = 0, t = 0; i < n; i++) {
        const Polinomio &e = s.ecs[i];
        for (int k = 0; k < e.num_terms; k++, t++) {
            polinomio_agregar_termino(&plano, t, e.coef[k], e.num_factores[k],
                                      e.var + k * MAX_FACTORES, e.exp + k * MAX_FACTORES);
            fila[t] = i;
        }
    }

    PolinomioGpu g;
    g.reservar(n, total);
    int *d_fila;
    double *d_x, *d_J;
    CUDA_OK(cudaMalloc(&d_fila, total * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_J, (size_t)n * n * sizeof(double)));

    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventCreate(&e3);
    CUDA_OK(cudaEventRecord(e0));
    g.copiar(plano, 0);
    CUDA_OK(cudaMemcpy(d_fila, fila.data(), total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_x, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaEventRecord(e1));
    CUDA_OK(cudaMemset(d_J, 0, (size_t)n * n * sizeof(double)));
    k_por_terminos<<<(total + bloque - 1) / bloque, bloque>>>(g.dev, d_fila, d_x, PASO_H_DEFECTO, d_J);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaEventRecord(e2));
    CUDA_OK(cudaMemcpy(J.data(), d_J, (size_t)n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaEventRecord(e3));
    CUDA_OK(cudaEventSynchronize(e3));
    float a, b, c;
    cudaEventElapsedTime(&a, e0, e1);
    cudaEventElapsedTime(&b, e1, e2);
    cudaEventElapsedTime(&c, e2, e3);

    printf("\n== Jacobiana de un sistema %d x %d (%d terminos por ecuacion) ==\n", n, n, terms);
    Referencias ref{-1.0, t_omp};
    registrar("jacobiana", "por_terminos", n, total, bloque, a, b, c, ref,
              error_relativo(J.data(), Jex.data(), n * n));

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(e3);
    cudaFree(d_fila);
    cudaFree(d_x);
    cudaFree(d_J);
    g.liberar();
    polinomio_liberar(&plano);
    sistema_liberar(&s);
}

static void ayuda() {
    puts("Uso: gradbench_cuda [opciones]\n"
         "  --vars N        variables del problema base (500)\n"
         "  --terms T       terminos (20000)\n"
         "  --bloques L     tamanos de bloque (32,64,128,256,512,1024)\n"
         "  --lista-vars L  variables para el experimento de escalado (500,2000,8000,32000)\n"
         "  --reps R        repeticiones por medida; se guarda la mediana (5)\n"
         "  --semilla S     semilla (2025)\n"
         "  --experimento E todo | precision (todo)\n"
         "  --csv FICHERO   guarda los resultados en CSV");
}

int main(int argc, char **argv) {
    Opciones o;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            ayuda();
            return 0;
        }
        if (i + 1 >= argc) {
            ayuda();
            return 2;
        }
        const char *v = argv[++i];
        if (!strcmp(a, "--vars")) o.vars = static_cast<int>(std::strtol(v, nullptr, 10));
        else if (!strcmp(a, "--terms")) o.terms = static_cast<int>(std::strtol(v, nullptr, 10));
        else if (!strcmp(a, "--bloques")) o.bloques = leer_lista(v);
        else if (!strcmp(a, "--lista-vars")) o.lista_vars = leer_lista(v);
        else if (!strcmp(a, "--reps")) o.reps = static_cast<int>(std::strtol(v, nullptr, 10));
        else if (!strcmp(a, "--semilla")) o.semilla = strtoull(v, nullptr, 10);
        else if (!strcmp(a, "--csv")) o.csv = v;
        else if (!strcmp(a, "--experimento")) o.experimento = v;
        else {
            ayuda();
            return 2;
        }
    }
    const bool bloques_validos = !o.bloques.empty() &&
        std::all_of(o.bloques.begin(), o.bloques.end(), [](int b) { return b >= 32 && b <= 1024; });
    if (o.experimento != "todo" && o.experimento != "precision") {
        fprintf(stderr, "Experimento desconocido: %s\n", o.experimento.c_str());
        return 2;
    }
    if (o.vars < 1 || o.terms < 1 || o.reps < 1 || !bloques_validos) {
        fprintf(stderr, "Valores fuera de rango (los bloques van de 32 a 1024 hilos)\n");
        return 2;
    }

    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (sm_%d%d, %d SM, %.1f GB) | CPU: %d nucleos logicos\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount, prop.totalGlobalMem / 1e9, omp_get_num_procs());
    if (prop.major < 6) {
        fprintf(stderr, "Hace falta una GPU con sm_60 o superior (atomicAdd en double)\n");
        return 1;
    }

    if (o.csv) {
        salida_csv = fopen(o.csv, "w");
        if (!salida_csv) {
            perror(o.csv);
            return 1;
        }
        fprintf(salida_csv, "experimento,kernel,vars,terminos,bloque,h2d_ms,kernel_ms,d2h_ms,total_ms,"
                            "speedup_vs_sec,speedup_vs_omp,error_rel\n");
    }

    if (o.experimento == "todo") {
        medir_problema("base", o, o.vars, true, o.bloques);
        for (int vars : o.lista_vars) {
            if (vars != o.vars) medir_problema("escalado_vars", o, vars, false, {128, 256});
        }
        medir_jacobiana(o);
    }
    medir_precision(o);

    if (salida_csv) fclose(salida_csv);
    return 0;
}
