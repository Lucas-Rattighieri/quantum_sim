import torch

from .config import device, dtype
from .bitops import *


def superposicao_uniforme(L, num_estados = 1) -> torch.Tensor:
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


def autoestado_z(L, i) -> torch.Tensor:
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


def autoestado_x(L, i) -> torch.Tensor:
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

    indice = gerar_indice(L)

    novo_indice = indice & i
    produto = 1 - 2 * (contar_bits(novo_indice, L) & 1)

    psi = produto / torch.sqrt(torch.tensor(2 ** L, device=device))

    return psi


def autoestado_y(L, i) -> torch.Tensor:
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

    indice = gerar_indice(L)
    bits_contados = contar_bits(indice, L)

    novo_indice = indice & i
    produto = (1 - 2 * ((contar_bits(novo_indice, L) ^ (bits_contados >> 1)) & 1)) * (1 + (bits_contados & 1) * (1j - 1))

    psi = produto / torch.sqrt(torch.tensor(2 ** L, dtype=dtype, device=device))

    return psi


def superposicao_hamming(L: int, d: int) -> torch.Tensor:
    """
    Gera um vetor de estado normalizado correspondente à superposição uniforme
    de todos os estados da base computacional com exatamente d bits 1 (peso de Hamming fixo).

    Parâmetros:
    - L (int): número total de qubits.
    - d (int): número de bits 1 (peso de Hamming).

    Retorna:
    - torch.Tensor: vetor de dimensão (2**L,) representando o estado quântico.
    """
    indice = gerar_indice(L)

    d_bits = contar_bits(indice, L) == d

    indice_d_bits = indice[d_bits]

    psi = torch.zeros(2**L, dtype=dtype, device=device)

    psi[indice_d_bits] = 1 / torch.sqrt(torch.tensor(len(indice_d_bits), dtype=dtype, device=device))

    return psi
