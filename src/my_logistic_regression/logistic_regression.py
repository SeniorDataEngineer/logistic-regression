#!/usr/bin/env python3.8.5
# Copyright 2020, Rose Software Ltd, All rights reserved.

# Built-in imports.
from math import log

# Project imports.
from has_property_mixin import HasPropertyMixin


class LogisticRegression(HasPropertyMixin):
    """
    This class can be used to perform logistic regression for binary
    classification. It is a supervised machine learning algorithm.
    """

    def __init__(self):
        """
        """
        pass

    def odds(
            self,
            p: float) -> float:
        """
        Odds returns the ratio of probability p to its inverse;
        not p. LT 1: p < np, 1: p = np and GT 1: p > np. \n
        Returns:
            float
        Doctest:
            >>> lr = LogisticRegression()
            >>> assert round(lr.odds(0.49), 3) == 0.961
            >>> assert lr.odds(0.5) == 1
            >>> assert round(lr.odds(0.51), 3) == 1.041
        """
        return p / (1 - p)

    def logit(
            self,
            p: float,
            f: str) -> float:
        """
        Logit returns the natural logarithm of the function of a
        probability p. Logit takes probabilities and returns a
        real between +inf and -inf. \n
        Returns:
            float
        Doctest:
            >>> lr = LogisticRegression()
            >>> assert round(lr.logit(p=0.6, f='odds'), 3) == 0.405
        """
        if not self.has_object(name=f):
            raise ValueError(f'The method {f} does not exist on class'
                            f'{self.__class__}.')
        f = getattr(self, f)
        return log(f(p))

    def sigmoid(
            self,
            z: float) -> float:
        """
        Sigmoid returns the logistic sigmoid for input z. The
        logistic sigmoid returns a probability of a real value
        between 0 and 1. The function's intercept is at 0.5. \n
        Returns:
            float
        Doctest:
            >>> lr = LogisticRegression()
            >>> assert round(lr.sigmoid(2), 3) == 0.881
        """
        e = 2.7182818284590452353602874713527
        return 1 / (1 + e ** (-z))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
