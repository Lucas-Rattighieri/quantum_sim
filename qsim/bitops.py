import torch

from .config import device


def gerar_indice(L):
    """
    Gera um tensor com os índices dos estados base de uma cadeia de L qubits.

    Parâmetros:
    - L (int): número de qubits.

    Retorna:
    - torch.Tensor: tensor de inteiros de 0 até 2^L - 1, no device padrão.
    """

    indice = torch.arange(2**L, dtype=torch.int32 if L <= 31 else torch.int64, device=device)
    return indice


def ligar_bit(num, p : int):
    """
    Define o bit na posição p como 1 no número num.

    Parâmetros:
    - num (int): número inteiro de entrada.
    - p (int): posição do bit a ser ativada.

    Retorna:
    - int: novo número com o bit p ligado (1).
    """
    return num | (1 << p)


def desligar_bit(num, p : int):
    """
    Define o bit na posição p como 0 no número num.

    Parâmetros:
    - num (int): número inteiro de entrada.
    - p (int): posição do bit a ser desativada.

    Retorna:
    - int: novo número com o bit p desligado (0).
    """
    return num & (~ (1 << p))


def bit(num, i : int):
    """
    Retorna o valor do bit na posição i do número num.

    Parâmetros:
    - num (int): número inteiro de entrada.
    - i (int): posição do bit a ser lida.

    Retorna:
    - int: valor do bit (0 ou 1).
    """
    return (num >> i) & 1


def contar_bits(num, L : int):
    """
    Conta o número de bits iguais a 1 nos L bits menos significativos de num.

    Parâmetros:
    - num (int): número inteiro de entrada.
    - L (int): número de bits considerados.

    Retorna:
    - int: quantidade de bits 1 entre os L bits menos significativos.
    """
    uns = 0
    for i in range(L):
        uns += (num >> i) & 1
    return uns


def permutar_bits(num, i : int, j : int):
    """
    Troca os bits nas posições i e j do número num.

    Parâmetros:
    - num (int): número inteiro de entrada.
    - i (int): primeira posição de bit.
    - j (int): segunda posição de bit.

    Retorna:
    - int: número com os bits i e j permutados (se forem diferentes).
    """
    mask = (((num >> i) ^ (num >> j)) & 1)
    return num ^ ((mask << i) | (mask << j))


def possui_um_bit_1(num: int):
    """
    Verifica se o número possui um bit 1 apenas.

    Parâmetros:
    - num (int): número inteiro de entrada.

    Retorna:
    - bool: Verificação se num possui um bit 1.
    """
    return (num != 0) & ((num & (num - 1)) == 0)


def translacao(num, d: int, L: int):
    """
    Aplica uma translação cíclica de d posições para a direita nos L bits de num.

    Parâmetros:
    - num (int): número inteiro de entrada.
    - d (int): número de posições a deslocar.
    - L (int): número total de bits considerados.

    Retorna:
    - int: número com os bits deslocados ciclicamente.
    """
    d %= L
    return (num >> (L - d)) | ((num << d) & ((1 << L) - 1))


def inversao(num, L: int):
    """
    Inverte todos os bits nos L bits menos significativos de num.

    Parâmetros:
    - num (int): número inteiro de entrada.
    - L (int): número de bits considerados.

    Retorna:
    - int: número com os L bits invertidos.
    """
    return (~ num) & ((1 << L) - 1)


def reflexao(num, L: int):
    """
    Reflete a ordem dos L bits de num (espelhamento da bitstring).

    Parâmetros:
    - num (int): número inteiro de entrada.
    - L (int): número total de bits.

    Retorna:
    - int: número com a ordem dos bits invertida (refletida).
    """
    n1 = 0
    for i in range(L):
        n1 |= ((num >> i) & 1) << (L - 1 - i)
    return n1
