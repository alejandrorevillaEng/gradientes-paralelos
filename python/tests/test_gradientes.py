"""Tests de los bindings de Python. Corren con unittest o con pytest."""

import doctest
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradientes  # noqa: E402
from gradientes import Polinomio, gradiente  # noqa: E402


def cerca(a, b, tol):
    return all(math.isclose(x, y, rel_tol=tol, abs_tol=tol) for x, y in zip(a, b, strict=True))


class TestPolinomio(unittest.TestCase):
    def setUp(self):
        # f(x, y) = 3x²y + 2y³  ->  ∇f = (6xy, 3x² + 6y²)
        self.f = Polinomio(2, [(3.0, [(0, 2), (1, 1)]), (2.0, [(1, 3)])])
        self.x = [1.0, 2.0]
        self.esperado = [12.0, 27.0]

    def tearDown(self):
        self.f.cerrar()

    def test_evaluar(self):
        self.assertAlmostEqual(self.f(self.x), 22.0)
        self.assertAlmostEqual(self.f(self.x, hilos=4), 22.0)

    def test_todos_los_metodos_coinciden(self):
        self.assertTrue(cerca(self.f.gradiente(self.x, metodo="exacto"), self.esperado, 1e-12))
        for metodo in ("terminos", "directo"):
            for hilos in (None, 1, 4):
                with self.subTest(metodo=metodo, hilos=hilos):
                    self.assertTrue(cerca(self.f.gradiente(self.x, metodo, hilos), self.esperado, 1e-6))

    def test_constante(self):
        with Polinomio(1, [(5.0, [])]) as c:
            self.assertEqual(c([3.0]), 5.0)
            self.assertEqual(c.gradiente([3.0], metodo="exacto"), [0.0])

    def test_aleatorio_reproducible_y_grande(self):
        p = Polinomio.aleatorio(200, 2000, semilla=7)
        q = Polinomio.aleatorio(200, 2000, semilla=7)
        x = [0.1 * ((i % 7) - 3) for i in range(200)]
        self.assertEqual(p(x), q(x))
        exacto = p.gradiente(x, metodo="exacto")
        self.assertTrue(cerca(p.gradiente(x, hilos=4), exacto, 1e-6))
        self.assertEqual(repr(p), "Polinomio(num_vars=200, num_terminos=2000)")

    def test_entradas_no_validas(self):
        with self.assertRaises(ValueError):
            self.f([1.0])
        with self.assertRaises(ValueError):
            self.f.gradiente(self.x, metodo="magia")
        with self.assertRaises(ValueError):
            self.f.gradiente(self.x, hilos=0)
        with self.assertRaises(ValueError):
            self.f.gradiente(self.x, h=0)
        with self.assertRaises(ValueError):
            Polinomio(2, [(1.0, [(2, 1)])])  # la variable 2 no existe
        with self.assertRaises(ValueError):
            Polinomio(2, [(1.0, [(0, -1)])])
        with self.assertRaises(ValueError):
            Polinomio(2, [(1.0, [(0, 1)] * 5)])
        with self.assertRaises(ValueError):
            Polinomio(0, [(1.0, [])])


class TestCajaNegra(unittest.TestCase):
    def test_rosenbrock(self):
        def rosenbrock(v):
            x, y = v
            return (1 - x) ** 2 + 100 * (y - x * x) ** 2

        g = gradiente(rosenbrock, [-1.2, 1.0])
        # ∇ = (-2(1-x) - 400x(y-x²), 200(y-x²))
        self.assertTrue(cerca(g, [-215.6, -88.0], 1e-5))

    def test_errores_de_la_funcion_llegan_a_python(self):
        def rota(_):
            raise ZeroDivisionError("a propósito")

        with self.assertRaises(ZeroDivisionError):
            gradiente(rota, [1.0, 2.0])


def load_tests(loader, tests, _patron):
    """El ejemplo del docstring del paquete también se ejecuta como test."""
    tests.addTests(doctest.DocTestSuite(gradientes))
    return tests


if __name__ == "__main__":
    unittest.main()
