import torch
from .. import diagonal as qd
from .. import bitops as qb
from .. import config as qconfig


def indice_maxcut(num_vertices : int, vertice_i : int, fixar_subconjunto : bool = False):
    """
    Retorna o índice da posição do bit associado ao vértice `vertice_i` na representação
    binária do estado para o problema MaxCut.

    Parâmetros:
        num_vertices (int): número total de vértices (qubits)
        vertice_i (int): índice do vértice (0 ≤ vertice_i < num_vertices)
        fixar_subconjunto (bool): se True, assume que o vértice 0 está fixado e não é codificado

    Retorna:
        int: índice do bit correspondente à variável binária associada ao vértice `vertice_i`.
             Retorna -1 se `vertice_i` estiver fixado (ou seja, vertice_i == 0 e fixar_subconjunto=True).
    """

    if fixar_subconjunto:
        if vertice_i == 0:
            return -1
        return num_vertices - vertice_i - 2
    else:
        return num_vertices - vertice_i - 1


def Hmaxcut(L: int, w, fixar_subconjunto : bool =False):
    """
    Constrói a diagonal do Hamiltoniano do MaxCut com codificação Z ∈ {±1}.

    Parâmetros:
        L (int): número de vértices
        w (Tensor): matriz de pesos (L x L)
        fixar_subconjunto (bool): se True, fixa o vértice 0 no subconjunto
            representado por 0

    Retorna:
        Tensor: vetor de dimensão 2^L (ou 2^(L - 1) se fixado) com a diagonal do Hamiltoniano
    """

    Ll = L - 1 if fixar_subconjunto else L
    
    H = torch.zeros(2**Ll, dtype=qconfig.dtype, device=qconfig.device)
    
    for i in range(L):
        for j in range(i + 1, L):  
            if w[i, j] != 0:
                ki = indice_maxcut(L, i, fixar_subconjunto)
                kj = indice_maxcut(L, j, fixar_subconjunto)
                if fixar_subconjunto:
                    if i == 0:
                        H += qd.cadeia_z(Ll, [kj], w[i, j])
                    else:
                        H += qd.cadeia_z(Ll, [ki, kj], w[i, j])
                else:
                    H += qd.cadeia_z(Ll, [ki, kj], w[i, j])

    return H


def particao_maxcut(num: int, L: int, fixar_subconjunto : bool = False):
    """
    Dado um inteiro que representa a configuração binária do MaxCut (como um estado base),
    retorna os subconjuntos A e B definidos por essa configuração.

    Parâmetros:
        num (int): inteiro representando a configuração de L bits (ou L - 1 se fixado)
        L (int): número total de vértices (qubits)
        fixar_subconjunto (bool): se True, fixa o vértice 0 em B e ignora seu bit na codificação

    Retorna:
        (list[int], list[int]): subconjuntos A e B com os índices dos vértices
    """

    A = []
    B = []

    if fixar_subconjunto:
        B.append(0)
        Ll = L - 1
        v = 1
    else:
        Ll = L
        v = 0

    for i in range(Ll):
        if qb.bit(num, i):
            B.append(i + v)
        else:
            A.append(i + v)

    return A, B
