import torch

def ligar_bit(n, p):
    return n | (1 << p)


def desligar_bit(n, p):
    return n & (~ (1 << p))


def bit(indice, i):
    return (indice >> i) & 1


def contar_bits(indice, L):
    uns = 0
    for i in range(L):
        uns += (indice >> i) & 1
    return uns


def translacao(num, d : int, L : int):
    d %= L
    return (num >> (L-d)) | ((num << d) & ((1 << L) - 1))


def inversao(num, L : int):
    return (~ num) & ((1 << L) -1)


def reflexao(num, L: int):
    n1 = 0
    for i in range(L):
        n1 |= ((num >> i) & 1) << (L - 1 - i)
    return n1