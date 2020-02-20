from __future__ import division


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    if b == 0:
        raise ZeroDivisionError('You cannot divide by 0')
    return a / b


def pow(a, b):
    return a ** b


def mod(a, b):
    return a % b
