/*
 * Resuelve el sistema de ejemplo del enunciado con Newton-Raphson, usando la
 * jacobiana calculada por diferencias finitas, y lo compara con la jacobiana
 * analítica.
 */
#include "gradiente.h"
#include "sistema.h"

#include <math.h>
#include <stdio.h>

static void imprimir_matriz(const double *J, int m, int n) {
    for (int i = 0; i < m; i++) {
        printf("    [");
        for (int j = 0; j < n; j++) printf(" %10.5f", J[i * n + j]);
        printf(" ]\n");
    }
}

int main(void) {
    Sistema s;
    if (sistema_ejemplo_enunciado(&s) != 0) {
        fprintf(stderr, "No se pudo crear el sistema\n");
        return 1;
    }
    puts("Sistema del enunciado:\n"
         "  3x^2 + yz + z + 5 = 2\n"
         "  2x + yz^3 + 2x + 2 = 0\n"
         "  3xz + 8yz + z^2 + 1 = 4\n");

    const double inicios[][3] = {{0.0, 0.0, -3.0}, {1.0, 1.0, -2.0}, {0.5, 0.0, -3.5}};
    const int num_inicios = (int)(sizeof(inicios) / sizeof(inicios[0]));
    int fallos = 0;

    for (int k = 0; k < num_inicios; k++) {
        double x[3] = {inicios[k][0], inicios[k][1], inicios[k][2]};
        printf("Desde (%.2f, %.2f, %.2f):\n", x[0], x[1], x[2]);

        double J[9], Jex[9];
        jacobiana_directa_sec(&s, x, PASO_H_DEFECTO, J);
        jacobiana_exacta(&s, x, Jex);
        puts("  Jacobiana por diferencias finitas en el punto inicial:");
        imprimir_matriz(J, 3, 3);
        printf("  Error relativo frente a la analitica: %.2e\n", error_relativo(J, Jex, 9));

        ResultadoNewton r = newton_resolver(&s, x, 100, 1e-12, PASO_H_DEFECTO, 1);
        if (r.convergio) {
            printf("  Solucion: x = %.8f, y = %.8f, z = %.8f (%d iteraciones, residuo %.1e)\n\n",
                   x[0], x[1], x[2], r.iteraciones, r.residuo);
        } else {
            printf("  No converge (residuo %.2e tras %d iteraciones)\n\n", r.residuo, r.iteraciones);
            fallos++;
        }
    }
    sistema_liberar(&s);
    return fallos == num_inicios ? 1 : 0;
}
