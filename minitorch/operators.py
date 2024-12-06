"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers. $f(x,y) = x * y$"""
    return x * y


def id(x: float) -> float:
    """Identity function. $f(x) = x$"""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers. $f(x,y) = x + y$"""
    return x + y


def neg(x: float) -> float:
    """Negate a number. $f(x) = -x$"""
    return -x


def lt(x: float, y: float) -> float:
    """$f(x) =$ 1.0 if x is less than y else 0.0"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """$f(x) =$ 1.0 if x is equal to y else 0.0"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """$f(x) = x$ if x greater than y else y"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """$f(x) = |x - y| < 1e-2$"""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Calculate the sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Caclulate the ReLU function.
    $f(x) = x$ if x is greater than 0, else 0.0.
    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculate the natural logarithm of x. $f(x) = log(x)$"""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculate the exponential of x. $f(x) = e^{x}$"""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculate the inverse of x. $f(x) = 1 /x$"""
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Calculate the gradient of the natural logarithm of x, multiplied by y."""
    return y / (x + EPS)


def inv_back(x: float, y: float) -> float:
    """Calculate the gradient of the inverse of x, multiplied by y."""
    return -(1.0 / x**2) * y


def relu_back(x: float, y: float) -> float:
    """Calculate the gradient of the ReLU function of x, multiplied by y."""
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addList : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher Order Map

    Args:
    ----
        fn (Callable[[float], float]): Function from one value to one value

    Returns:
    -------
        Callable[[Iterable[float]], Iterable[float]]: A function that takes a lift, applies `fn` to each element, and returns a new list.

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        return [fn(i) for i in ls]

    return _map


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Use `map` to negate each element in `ls`."""
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher order zipwith

    Args:
    ----
        fn (Callable[[float, float], float]): combine two values into one

    Returns:
    -------
        Callable[[Iterable[float], Iterable[float]], Iterable[float]]: Function that takes two equal length lists `ls1` and `ls2`, applies `fn` to each pair of elements, and returns a new list.

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return _zipWith


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements of `ls1` and `ls2` using `zipWith` and `add`."""
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order reduce

    Args:
    ----
        fn (Callable[[float, float], float]): combine two values
        start (float): start value $x_0$

    Returns:
    -------
        Callable[[Iterable[float]], float]: Function that takes a list `ls` and computes the reduction :math:`fn(x_3, fn(x_2, fn(x_1, x_0)))`

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def sum(x: Iterable[float]) -> float:
    """Sum up a list using `reduce` and `add`."""
    return reduce(add, 0.0)(x)


def prod(x: Iterable[float]) -> float:
    """Take the product of a list using `reduce` and `mul`."""
    return reduce(mul, 1.0)(x)
