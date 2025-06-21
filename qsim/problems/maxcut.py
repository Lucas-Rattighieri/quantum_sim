from .. import diagonal as qd
from .. import bitops as qb
from .. import config as qconfig

def Hmaxcut(L : int, w):
    """
    Constrói a diagonal do Hamiltoniano correspondente ao problema MaxCut,
    codificado como um problema QUBO com variáveis binárias z_i ∈ {±1}.

    Parâmetros:
        L (int): número de qubits (ou vértices do grafo)
        w (Tensor): matriz de pesos w_{ij} das arestas do grafo (L x L)

    Retorna:
        torch.Tensor: vetor de dimensão 2^L representando a diagonal do Hamiltoniano,
                      com as contribuições de cada estado base.
    """
  
    H = torch.zeros(2**L, dtype= qconfig.dtype, device= qconfig.device)

    for i in range(L):
        for j in range(L):
            if (w[i,j] != 0):
                termo = qd.cadeia_z(L, [i, j], w[i, j] / 2)
                H = H + termo
    return H

def particao_maxcut(num: int, L: int):
    """
    Transcreve um inteiro num (representando uma configuração binária) em dois subconjuntos
    correspondentes à partição do grafo no MaxCut, com x_i ∈ {0,1}.

    Parâmetros:
        num (int): inteiro representando a bitstring da configuração
        L (int): número de vértices (bits)

    Retorna:
        (list[int], list[int]): tupla com os vértices em cada subconjunto
    """
    A = []
    B = []

    for i in range(L):
        if qb.bit(num, i):
            B.append(i)
        else:
            A.append(i)

    return A, B
