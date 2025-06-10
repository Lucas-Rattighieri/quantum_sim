import torch
from .operators import Z, X, Y, ZZ, XX, YY



def Hx(psi : torch.Tensor, L : int, indice : torch.Tensor):
    """
    Aplica a ação do Hamiltoniano X sobre o vetor de estado `psi`.

    A função computa a contribuição de termos locais do tipo X_i para cada sítio i,
    somando suas ações no vetor de estado. O resultado é equivalente à aplicação do Hamiltoniano:

        H_X = ∑_{i=0}^{L-1} X_i

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - indice (torch.Tensor): tensor com os índices inteiros correspondentes aos estados base.

    Retorna:
    - Hxpsi (torch.Tensor): vetor resultante da aplicação do Hamiltoniano X.
    """

    Hxpsi = torch.zeros_like(psi)

    for i in range(L):
        Hxpsi += X(psi, L, i, indice)
    return Hxpsi


def Hy(psi : torch.Tensor, L : int, indice : torch.Tensor):
    """
    Aplica a ação do Hamiltoniano Y sobre o vetor de estado `psi`.

    A função computa a contribuição de termos locais do tipo Y_i para cada sítio i,
    somando suas ações no vetor de estado. O resultado é equivalente à aplicação do Hamiltoniano:

        H_Z = ∑_{i=0}^{L-1} Y_i

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - indice (torch.Tensor): tensor com os índices inteiros correspondentes aos estados base.

    Retorna:
    - Hypsi (torch.Tensor): vetor resultante da aplicação do Hamiltoniano Y.
    """

    Hypsi = torch.zeros_like(psi)

    for i in range(L):
        Hypsi += Y(psi, L, i, indice)
    return Hypsi


def Hz(psi : torch.Tensor, L : int, indice : torch.Tensor):
    """
    Aplica a ação do Hamiltoniano Z sobre o vetor de estado `psi`.

    A função computa a contribuição de termos locais do tipo Z_i para cada sítio i,
    somando suas ações no vetor de estado. O resultado é equivalente à aplicação do Hamiltoniano:

        H_Z = ∑_{i=0}^{L-1} Z_i

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - indice (torch.Tensor): tensor com os índices inteiros correspondentes aos estados base.

    Retorna:
    - Hzpsi (torch.Tensor): vetor resultante da aplicação do Hamiltoniano Z.
    """

    Hzpsi = torch.zeros_like(psi)

    for i in range(L):
        Hzpsi += Z(psi, L, i, indice)
    return Hzpsi


def Hxx(psi : torch.Tensor, L : int, w, indice : torch.Tensor):
    """
    Aplica a ação do Hamiltoniano XX sobre o vetor de estado `psi`.

    A função computa a contribuição de termos do tipo X_i X_j para todos os pares (i < j)
    com acoplamento indicado por w[i, j] == 1. O resultado é a aplicação total de:

        H_XX = ∑_{i<j} w[i,j] * X_i X_j

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - w (array ou tensor): matriz simétrica indicando os acoplamentos. w[i, j] deve ser 1 (ativa interação) ou 0 (sem interação).
    - indice (torch.Tensor): tensor com os índices inteiros correspondentes aos estados base.

    Retorna:
    - Hxxpsi (torch.Tensor): vetor resultante da aplicação do Hamiltoniano XX.
    """

    Hxxpsi = torch.zeros_like(psi)

    for i in range(L):
        for j in range(i + 1, L):
            if w[i, j] != 0:
                Hxxpsi += XX(psi, L, i, j, indice)

    return Hxxpsi


def Hyy(psi : torch.Tensor, L : int, w, indice : torch.Tensor):
    """
    Aplica a ação do Hamiltoniano YY sobre o vetor de estado `psi`.

    A função computa a contribuição de termos do tipo Y_i Y_j para todos os pares (i < j)
    com acoplamento indicado por w[i, j] == 1. O resultado é a aplicação total de:

        H_YY = ∑_{i<j} w[i,j] * Y_i Y_j

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - w (array ou tensor): matriz simétrica indicando os acoplamentos. w[i, j] deve ser 1 (ativa interação) ou 0 (sem interação).
    - indice (torch.Tensor): tensor com os índices inteiros correspondentes aos estados base.

    Retorna:
    - Hyypsi (torch.Tensor): vetor resultante da aplicação do Hamiltoniano YY.
    """

    Hyypsi = torch.zeros_like(psi)

    for i in range(L):
        for j in range(i + 1, L):
            if w[i, j] != 0:
                Hyypsi += YY(psi, L, i, j, indice)
    
    return Hyypsi


def Hzz(psi : torch.Tensor, L : int, w, indice : torch.Tensor):
    """
    Aplica a ação do Hamiltoniano ZZ sobre o vetor de estado `psi`.

    A função computa a contribuição de termos do tipo Z_i Z_j para todos os pares (i < j)
    com acoplamento indicado por w[i, j] == 1. O resultado é a aplicação total de:

        H_ZZ = ∑_{i<j} w[i,j] * Z_i Z_j

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - w (array ou tensor): matriz simétrica indicando os acoplamentos. w[i, j] deve ser 1 (ativa interação) ou 0 (sem interação).
    - indice (torch.Tensor): tensor com os índices inteiros correspondentes aos estados base.

    Retorna:
    - Hzzpsi (torch.Tensor): vetor resultante da aplicação do Hamiltoniano ZZ.
    """

    Hzzpsi = torch.zeros_like(psi)

    for i in range(L):
        for j in range(i + 1, L):
            if w[i, j] != 0:
                Hzzpsi += ZZ(psi, L, i, j, indice)
    
    return Hzzpsi


def Hxy(psi : torch.Tensor, L : int, w, indice : torch.Tensor):
    """
    Aplica a ação do Hamiltoniano XY sobre o vetor de estado `psi`.

    A função computa a contribuição de termos do tipo (X_i X_j + Y_i Y_j) para todos os pares (i < j)
    com acoplamento indicado por w[i, j] == 1. A operação resulta na troca dos estados |01⟩ e |10⟩ 
    nos sítios i e j, preservando a paridade total. O resultado é a aplicação total de:

        H_XY = ∑_{i<j} w[i,j] * (X_i X_j + Y_i Y_j)

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - w (array ou tensor): matriz simétrica indicando os acoplamentos. w[i, j] deve ser 1 (ativa interação) ou 0 (sem interação).
    - indice (torch.Tensor): tensor com os índices inteiros correspondentes aos estados base.

    Retorna:
    - Hxypsi (torch.Tensor): vetor resultante da aplicação do Hamiltoniano XY.
    """

    Hxypsi = torch.zeros_like(psi)

    for i in range(L):
        for j in range(i+1, L):
            if w[i, j] != 0:
                mask01 = (((indice >> i) ^ (indice >> j)) & 1) == 1
                flip = (1 << i) | (1 << j)

                indices01 = indice[mask01]
                indices10 = indices01 ^ flip

                Hxypsi[indices01] += psi[indices10]

    return Hxypsi

