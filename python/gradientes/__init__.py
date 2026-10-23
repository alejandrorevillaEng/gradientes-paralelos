"""Gradientes de polinomios multivariable en C con OpenMP, desde Python.

Envuelve con ctypes la biblioteca compartida que compila CMake
(libgradientes.so, libgradientes.dylib o gradientes.dll). La busca en
la variable de entorno GRADIENTES_LIB o en las carpetas build*/ del repo.

    >>> from gradientes import Polinomio
    >>> f = Polinomio(2, [(3.0, [(0, 2), (1, 1)]), (2.0, [(1, 3)])])  # 3x²y + 2y³
    >>> f([1.0, 2.0])
    22.0
    >>> [round(g, 6) for g in f.gradiente([1.0, 2.0])]
    [12.0, 27.0]
"""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

__all__ = ["Polinomio", "gradiente", "MAX_FACTORES", "PASO_H"]

MAX_FACTORES = 4
PASO_H = 1e-6

_METODOS = ("terminos", "directo", "exacto")


class _PolinomioC(ctypes.Structure):
    _fields_ = [
        ("num_vars", ctypes.c_int),
        ("num_terms", ctypes.c_int),
        ("coef", ctypes.POINTER(ctypes.c_double)),
        ("num_factores", ctypes.POINTER(ctypes.c_int)),
        ("var", ctypes.POINTER(ctypes.c_int)),
        ("exp", ctypes.POINTER(ctypes.c_int)),
    ]


class _RngC(ctypes.Structure):
    _fields_ = [("estado", ctypes.c_uint64)]


_P = ctypes.POINTER(_PolinomioC)
_D = ctypes.POINTER(ctypes.c_double)
_I = ctypes.POINTER(ctypes.c_int)
_FuncionEscalar = ctypes.CFUNCTYPE(ctypes.c_double, _D, ctypes.c_void_p)


def _nombre_biblioteca() -> str:
    if sys.platform == "win32":
        return "gradientes.dll"
    if sys.platform == "darwin":
        return "libgradientes.dylib"
    return "libgradientes.so"


def _buscar_biblioteca() -> Path:
    ruta = os.environ.get("GRADIENTES_LIB")
    if ruta:
        return Path(ruta)
    raiz = Path(__file__).resolve().parents[2]
    nombre = _nombre_biblioteca()
    for carpeta in sorted(raiz.glob("build*")):
        for candidata in carpeta.rglob(nombre):
            return candidata
    raise OSError(
        f"No encuentro {nombre}. Compílala con 'cmake --preset release && cmake --build --preset release' "
        "o indica la ruta en GRADIENTES_LIB."
    )


def _cargar() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_buscar_biblioteca()))
    firmas = {
        "polinomio_crear": (ctypes.c_int, [_P, ctypes.c_int, ctypes.c_int]),
        "polinomio_liberar": (None, [_P]),
        "polinomio_agregar_termino": (ctypes.c_int, [_P, ctypes.c_int, ctypes.c_double, ctypes.c_int, _I, _I]),
        "polinomio_aleatorio": (ctypes.c_int, [_P, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                               ctypes.c_double, ctypes.POINTER(_RngC)]),
        "rng_iniciar": (None, [ctypes.POINTER(_RngC), ctypes.c_uint64]),
        "polinomio_evaluar": (ctypes.c_double, [_P, _D]),
        "polinomio_evaluar_omp": (ctypes.c_double, [_P, _D, ctypes.c_int]),
        "polinomio_gradiente_exacto": (None, [_P, _D, _D]),
        "grad_directo_sec": (None, [_P, _D, ctypes.c_double, _D]),
        "grad_directo_omp": (None, [_P, _D, ctypes.c_double, _D, ctypes.c_int]),
        "grad_por_terminos_sec": (None, [_P, _D, ctypes.c_double, _D]),
        "grad_por_terminos_omp": (None, [_P, _D, ctypes.c_double, _D, ctypes.c_int]),
        "gradiente_fd_sec": (None, [_FuncionEscalar, ctypes.c_void_p, _D, ctypes.c_int, ctypes.c_double, _D]),
    }
    for nombre, (resultado, argumentos) in firmas.items():
        funcion = getattr(lib, nombre)
        funcion.restype = resultado
        funcion.argtypes = argumentos
    return lib


_lib = _cargar()


def _vector(x: Iterable[float], n: int) -> ctypes.Array:
    valores = [float(v) for v in x]
    if len(valores) != n:
        raise ValueError(f"El punto tiene {len(valores)} coordenadas y el polinomio {n} variables")
    return (ctypes.c_double * n)(*valores)


def _comprobar_hilos(hilos: int | None) -> int | None:
    if hilos is not None and (not isinstance(hilos, int) or hilos < 1):
        raise ValueError("hilos tiene que ser un entero mayor que 0")
    return hilos


class Polinomio:
    """Polinomio multivariable. Cada término es (coeficiente, [(variable, exponente), ...])."""

    def __init__(self, num_vars: int, terminos: Sequence[tuple[float, Sequence[tuple[int, int]]]]):
        if num_vars < 1 or not terminos:
            raise ValueError("Hace falta al menos una variable y un término")
        self._c = _PolinomioC()
        if _lib.polinomio_crear(ctypes.byref(self._c), num_vars, len(terminos)) != 0:
            raise MemoryError("No se pudo reservar el polinomio")
        for t, (coef, factores) in enumerate(terminos):
            factores = list(factores)
            if len(factores) > MAX_FACTORES:
                self.cerrar()
                raise ValueError(f"El término {t} tiene más de {MAX_FACTORES} factores")
            nf = len(factores)
            variables = (ctypes.c_int * max(nf, 1))(*[v for v, _ in factores])
            exponentes = (ctypes.c_int * max(nf, 1))(*[e for _, e in factores])
            if _lib.polinomio_agregar_termino(ctypes.byref(self._c), t, float(coef), nf, variables, exponentes) < 0:
                self.cerrar()
                raise ValueError(f"Término {t} no válido: variables en [0, {num_vars}) y exponentes >= 1")

    @classmethod
    def aleatorio(cls, num_vars: int, num_terminos: int, max_factores: int = 3, max_exp: int = 3,
                  rango_coef: float = 1.0, semilla: int = 1) -> Polinomio:
        """El mismo polinomio aleatorio que genera el benchmark de C para esa semilla."""
        nuevo = cls.__new__(cls)
        nuevo._c = _PolinomioC()
        rng = _RngC()
        _lib.rng_iniciar(ctypes.byref(rng), semilla)
        if _lib.polinomio_aleatorio(ctypes.byref(nuevo._c), num_vars, num_terminos, max_factores, max_exp,
                                    rango_coef, ctypes.byref(rng)) != 0:
            raise ValueError("Parámetros no válidos para el polinomio aleatorio")
        return nuevo

    @property
    def num_vars(self) -> int:
        return self._c.num_vars

    @property
    def num_terminos(self) -> int:
        return self._c.num_terms

    def __call__(self, x: Iterable[float], hilos: int | None = None) -> float:
        v = _vector(x, self.num_vars)
        if _comprobar_hilos(hilos) is None:
            return _lib.polinomio_evaluar(ctypes.byref(self._c), v)
        return _lib.polinomio_evaluar_omp(ctypes.byref(self._c), v, hilos)

    def gradiente(self, x: Iterable[float], metodo: str = "terminos", hilos: int | None = None,
                  h: float = PASO_H) -> list[float]:
        """Gradiente en x.

        metodo: "terminos" (diferencias centradas término a término, O(términos)),
        "directo" (2n evaluaciones del polinomio, el método de la práctica) o "exacto"
        (derivada analítica). Con hilos=None se usa la versión secuencial.
        """
        if metodo not in _METODOS:
            raise ValueError(f"metodo tiene que ser uno de {_METODOS}")
        if h <= 0:
            raise ValueError("h tiene que ser positivo")
        _comprobar_hilos(hilos)
        n = self.num_vars
        v = _vector(x, n)
        g = (ctypes.c_double * n)()
        p = ctypes.byref(self._c)
        if metodo == "exacto":
            _lib.polinomio_gradiente_exacto(p, v, g)
        elif metodo == "terminos":
            if hilos is None:
                _lib.grad_por_terminos_sec(p, v, h, g)
            else:
                _lib.grad_por_terminos_omp(p, v, h, g, hilos)
        elif hilos is None:
            _lib.grad_directo_sec(p, v, h, g)
        else:
            _lib.grad_directo_omp(p, v, h, g, hilos)
        return list(g)

    def cerrar(self) -> None:
        if getattr(self, "_c", None) is not None and self._c.coef:
            _lib.polinomio_liberar(ctypes.byref(self._c))

    def __del__(self) -> None:
        self.cerrar()

    def __enter__(self) -> Polinomio:
        return self

    def __exit__(self, *_: object) -> None:
        self.cerrar()

    def __repr__(self) -> str:
        return f"Polinomio(num_vars={self.num_vars}, num_terminos={self.num_terminos})"


def gradiente(f: Callable[[list[float]], float], x: Sequence[float], h: float = PASO_H) -> list[float]:
    """Gradiente "caja negra" de cualquier función de Python con el código de C.

    Usa diferencias centradas (2n llamadas a f). Es secuencial: una función de Python
    no se puede llamar desde varios hilos de OpenMP a la vez.
    """
    if h <= 0:
        raise ValueError("h tiene que ser positivo")
    n = len(x)
    if n < 1:
        raise ValueError("El punto tiene que tener al menos una coordenada")
    error: list[BaseException] = []

    def adaptador(punto: ctypes.POINTER(ctypes.c_double), _ctx: int) -> float:
        try:
            return float(f([punto[i] for i in range(n)]))
        except BaseException as e:  # noqa: BLE001 - se relanza al volver a Python
            error.append(e)
            return 0.0

    callback = _FuncionEscalar(adaptador)
    g = (ctypes.c_double * n)()
    _lib.gradiente_fd_sec(callback, None, _vector(x, n), n, h, g)
    if error:
        raise error[0]
    return list(g)
