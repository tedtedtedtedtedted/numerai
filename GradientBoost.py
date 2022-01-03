# Implement Strategy Design Pattern.
# An abstract superclass for many variants of the Gradient Descent (GD) algorithm.


from typing import List, Union
import numpy as np

from RegressionModel import RegressionModel


class GradientDescent:
    """
    The abstract super class for many variants of the Gradient Descent (GD) algorithm.

    The goal is to optimize parameter for models (minimizes the cost function).
    """
    def __init__(self, parameters: List[Union[int, float]] = None) -> None:
        """
        Accept (potentially) many parameters, and initialize them accordingly.
        TODO: Perhaps make the type to be list of objects for generality? Or
              maybe even array? Not sure.
        """
        pass

    def __str__(self) -> str:
        """
        Return a string representation/characterization of this GD algorithm.
        Output format:
                      GD_VARIANT_NAME
                      Parameter 1: PARAM_1
                      Parameter 2: PARAM_2
                         ...     :  ...
                         ...     :  ...
                         ...     :  ...
        """
        pass

    def optimize(self, model: RegressionModel) -> np.ndarray[float]:
        """
        TODO: Temporarily assume the model GD applied on must be "RegressionModel"

        Return a NumPy N-dim arary object as the optimized result of parameters.
        """
        pass



