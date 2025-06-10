import torch

from .config import device, dtype
from .bitops import contar_bits


def superposicao_uniforme(L, num_estados = 1):
    """
    Gera um vetor de estado representando uma superposição uniforme sobre todos os estados base de L qubits.

    Se num_estados > 1, gera múltiplos vetores de superposição uniforme em colunas.

    Parâmetros:
    - L (int): número de qubits.
    - num_estados (int, opcional): número de vetores de estado a serem gerados. Default é 1.

    Retorna:
    - psi (torch.Tensor): vetor(s) de estado(s) com amplitudes iguais para todos os estados base,
    normalizados para norma 1.
    """

    if num_estados == 1:
        psi = torch.ones(2**L, dtype=dtype, device=device)
    else:
        psi = torch.ones(2 ** L, num_estados, dtype=dtype, device=device)

    psi /= torch.sqrt(torch.tensor(2 ** L, device=device))

    return psi


def autoestado_z(L, i):
    """
    Gera o autoestado do operador soma dos Z, ∑_a Z_a, correspondente ao estado base computacional |i⟩.

    Parâmetros:
    - L (int): número total de qubits.
    - i (int): inteiro representando o estado base |i⟩.

    Retorna:
    - psi (torch.Tensor): vetor de estado com 1 na posição i e zero nas demais, representando |i⟩.
    """

    psi = torch.zeros(2**L, dtype=dtype, device=device)
    psi[i] = 1
    return psi


def autoestado_x(L, i):
    """
    Gera um autoestado do operador soma dos X, ∑_a X_a, para um sistema de L qubits.

    O parâmetro `i` representa uma bitstring na base {|+⟩ = (|0⟩ + |1⟩)/2, |-⟩ = (|0⟩ - |1⟩)/2}, onde (+) é codificado como 0 
    e (-) como 1 (Ex: i = 3, L = 5 -> |+++--⟩). 

    Parâmetros:
    - L (int): número total de qubits.
    - i (int): inteiro representando a bitstring na base {|+⟩, |-⟩}.

    Retorna:
    - psi (torch.Tensor): vetor de estado normalizado correspondente ao autoestado do operador ∑_a X_a.
    """

    indice = torch.arange(2**L, dtype=torch.int64, device=device)

    novo_indice = indice & i
    produto = 1 - 2 * (contar_bits(novo_indice, L) & 1)

    psi = produto / torch.sqrt(torch.tensor(2 ** L, device=device))

    return psi


def autoestado_y(L, i):
    """
    Gera um autoestado do operador soma dos Y, ∑_a Y_a, para um sistema de L qubits.

    O parâmetro `i` representa uma bitstring na base {|+i⟩ = (|0⟩ + i|1⟩)/2, |-i⟩ = (|0⟩ - i|1⟩)/2}, onde (+i) é codificado como 0 
    e (-i) como 1 (Ex: i = 3, L = 5 -> |(+i)(+i)(+i)(-i)(-i)⟩). 

    Parâmetros:
    - L (int): número total de qubits.
    - i (int): inteiro representando a bitstring na base {|+i⟩, |-i⟩}.

    Retorna:
    - psi (torch.Tensor): vetor de estado normalizado correspondente ao autoestado do operador ∑_a Y_a.
    """

    indice = torch.arange(2**L, dtype=torch.int64)
    bits_contados = contar_bits(indice, L)

    novo_indice = indice & i
    produto = produto = (1 - 2 * ((contar_bits(novo_indice, L) ^ (bits_contados >> 1)) & 1)) * (1 + (bits_contados & 1) * (1j - 1))

    psi = produto / torch.sqrt(torch.tensor(2 ** L))

    return psi
