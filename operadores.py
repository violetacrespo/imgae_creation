# -*- coding: utf-8 -*-
# operadores.py

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.mutation.gauss import GaussianMutation


def get_crossover(operator_name, **kwargs):
    if operator_name == "sbx":
        return SBX(prob=kwargs.get("prob", 0.9), eta=kwargs.get("eta", 15))
    elif operator_name == "uniform":
        return UniformCrossover()
    else:
        raise ValueError(f"Operador de cruce no reconocido: {operator_name}")


def get_mutation(operator_name, **kwargs):
    if operator_name == "polynomial":
        return PolynomialMutation(prob=kwargs.get("prob", 0.2), eta=kwargs.get("eta", 20))
    elif operator_name == "gaussian":
        return GaussianMutation(sigma=kwargs.get("sigma", 0.1), prob=kwargs.get("prob", 0.2))
    else:
        raise ValueError(f"Operador de mutación no reconocido: {operator_name}")
