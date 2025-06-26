import torch
from .operators import Z, X, Y, ZZ, XX, YY
from .bitops import gerar_indice


def Hx(psi : torch.Tensor, L : int, 
       indice : torch.Tensor = None, tmp: torch.Tensor = None, 
       out: torch.Tensor = None, tmppsi: torch.Tensor = None):
    """
    Aplica a ação do Hamiltoniano X sobre o vetor de estado `psi`.

    A função computa a contribuição de termos locais do tipo X_i para cada sítio i,
    somando suas ações no vetor de estado. O resultado é equivalente à aplicação do Hamiltoniano:

        H_X = ∑_{i=0}^{L-1} X_i

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros dos estados base.
    - tmp (torch.Tensor, opcional): tensor auxiliar para armazenar os índices modificados.
    - out (torch.Tensor, opcional): vetor de saída para o resultado (evita alocação).
    - tmppsi (torch.Tensor, opcional): tensor auxiliar complexo.

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação do Hamiltoniano X.
    """

    if indice is None:
        indice = gerar_indice(L)

    if tmppsi is None:
        tmppsi.empty_like(psi)

    if out is None:
        out = torch.zeros_like(psi)
    else:
        out.zeros_()

    for i in range(L):
        tmppsi = X(psi, L, i, indice, tmp, tmppsi)
        out.add_(tmppsi)
    return out


def Hy(psi : torch.Tensor, L : int, 
       indice : torch.Tensor = None, tmp: torch.Tensor = None, 
       out: torch.Tensor = None, tmppsi: torch.Tensor = None):
    """
    Aplica a ação do Hamiltoniano Y sobre o vetor de estado `psi`.

    A função computa a contribuição de termos locais do tipo Y_i para cada sítio i,
    somando suas ações no vetor de estado. O resultado é equivalente à aplicação do Hamiltoniano:

        H_Z = ∑_{i=0}^{L-1} Y_i

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação do Hamiltoniano Y.
    """

    if indice is None:
        indice = gerar_indice(L)

    if tmppsi is None:
        tmppsi.empty_like(psi)

    if out is None:
        out = torch.zeros_like(psi)
    else:
        out.zeros_()

    for i in range(L):
        tmppsi = Y(psi, L, i, indice, tmp, tmppsi)
        out.add_(tmppsi)
    return out


def Hz(psi : torch.Tensor, L : int, 
       indice : torch.Tensor = None, tmp: torch.Tensor = None, 
       out: torch.Tensor = None, tmppsi: torch.Tensor = None):
    """
    Aplica a ação do Hamiltoniano Z sobre o vetor de estado `psi`.

    A função computa a contribuição de termos locais do tipo Z_i para cada sítio i,
    somando suas ações no vetor de estado. O resultado é equivalente à aplicação do Hamiltoniano:

        H_Z = ∑_{i=0}^{L-1} Z_i

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros dos estados base.
    - tmp (torch.Tensor, opcional): tensor auxiliar para armazenar os índices modificados.
    - out (torch.Tensor, opcional): vetor de saída para o resultado (evita alocação).
    - (torch.Tensor, opcional): tensor auxiliar complexo.

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação do Hamiltoniano Z.
    """

    if indice is None:
        indice = gerar_indice(L)

    if tmppsi is None:
        tmppsi.empty_like(psi)

    if out is None:
        out = torch.zeros_like(psi)
    else:
        out.zeros_()

    for i in range(L):
        tmppsi = Z(psi, L, i, indice, tmp, tmppsi)
        out.add_(tmppsi)
    return out


def Hxx(psi : torch.Tensor, L : int, w, 
       indice : torch.Tensor = None, tmp: torch.Tensor = None, 
       out: torch.Tensor = None, tmppsi: torch.Tensor = None):
    """
    Aplica a ação do Hamiltoniano XX sobre o vetor de estado `psi`.

    A função computa a contribuição de termos do tipo X_i X_j para todos os pares (i < j)
    com acoplamento indicado por w[i, j] == 1. O resultado é a aplicação total de:

        H_XX = ∑_{i<j} w[i,j] * X_i X_j

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - w (array ou tensor): matriz simétrica indicando os acoplamentos. w[i, j] deve ser 1 (ativa interação) ou 0 (sem interação).
    - indice (torch.Tensor, opcional): tensor com os índices inteiros dos estados base.
    - tmp (torch.Tensor, opcional): tensor auxiliar para armazenar os índices modificados.
    - out (torch.Tensor, opcional): vetor de saída para o resultado (evita alocação).
    - (torch.Tensor, opcional): tensor auxiliar complexo.

    Retorna:
    - Hxxpsi (torch.Tensor): vetor resultante da aplicação do Hamiltoniano XX.
    """

    if indice is None:
        indice = gerar_indice(L)

    if tmppsi is None:
        tmppsi.empty_like(psi)

    if out is None:
        out = torch.zeros_like(psi)
    else:
        out.zeros_()

    for i in range(L):
        for j in range(i + 1, L):
            if w[i, j] != 0:
                tmppsi = XX(psi, L, i, j, indice, tmp, tmppsi)
                out.add_(tmppsi)
    return out


def Hyy(psi : torch.Tensor, L : int, w, 
       indice : torch.Tensor = None, tmp: torch.Tensor = None, 
       out: torch.Tensor = None, tmppsi: torch.Tensor = None):
    """
    Aplica a ação do Hamiltoniano YY sobre o vetor de estado `psi`.

    A função computa a contribuição de termos do tipo Y_i Y_j para todos os pares (i < j)
    com acoplamento indicado por w[i, j] == 1. O resultado é a aplicação total de:

        H_YY = ∑_{i<j} w[i,j] * Y_i Y_j

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - w (array ou tensor): matriz simétrica indicando os acoplamentos. w[i, j] deve ser 1 (ativa interação) ou 0 (sem interação).
    - indice (torch.Tensor, opcional): tensor com os índices inteiros dos estados base.
    - tmp (torch.Tensor, opcional): tensor auxiliar para armazenar os índices modificados.
    - out (torch.Tensor, opcional): vetor de saída para o resultado (evita alocação).
    - (torch.Tensor, opcional): tensor auxiliar complexo.

    Retorna:
    - Hyypsi (torch.Tensor): vetor resultante da aplicação do Hamiltoniano YY.
    """

    if indice is None:
        indice = gerar_indice(L)

    if tmppsi is None:
        tmppsi.empty_like(psi)

    if out is None:
        out = torch.zeros_like(psi)
    else:
        out.zeros_()

    for i in range(L):
        for j in range(i + 1, L):
            if w[i, j] != 0:
                tmppsi = YY(psi, L, i, j, indice, tmp, tmppsi)
                out.add_(tmppsi)
    return out




def Hzz(psi : torch.Tensor, L : int, w, 
       indice : torch.Tensor = None, tmp: torch.Tensor = None, 
       out: torch.Tensor = None, tmppsi: torch.Tensor = None):
    """
    Aplica a ação do Hamiltoniano ZZ sobre o vetor de estado `psi`.

    A função computa a contribuição de termos do tipo Z_i Z_j para todos os pares (i < j)
    com acoplamento indicado por w[i, j] == 1. O resultado é a aplicação total de:

        H_ZZ = ∑_{i<j} w[i,j] * Z_i Z_j

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - w (array ou tensor): matriz simétrica indicando os acoplamentos. w[i, j] deve ser 1 (ativa interação) ou 0 (sem interação).
    - indice (torch.Tensor, opcional): tensor com os índices inteiros dos estados base.
    - tmp (torch.Tensor, opcional): tensor auxiliar para armazenar os índices modificados.
    - out (torch.Tensor, opcional): vetor de saída para o resultado (evita alocação).
    - (torch.Tensor, opcional): tensor auxiliar complexo.

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação do Hamiltoniano ZZ.
    """

    if indice is None:
        indice = gerar_indice(L)

    if tmppsi is None:
        tmppsi.empty_like(psi)

    if out is None:
        out = torch.zeros_like(psi)
    else:
        out.zeros_()

    for i in range(L):
        for j in range(i + 1, L):
            if w[i, j] != 0:
                tmppsi = ZZ(psi, L, i, j, indice, tmp, tmppsi)
                out.add_(tmppsi)
    return out


def Hxy(psi : torch.Tensor, L : int, w, 
       indice : torch.Tensor = None, tmp: torch.Tensor = None, 
       out: torch.Tensor = None, tmppsi: torch.Tensor = None):
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
    - indice (torch.Tensor, opcional): tensor com os índices inteiros dos estados base.
    - tmp (torch.Tensor, opcional): tensor auxiliar para armazenar os índices modificados.
    - out (torch.Tensor, opcional): vetor de saída para o resultado (evita alocação).
    - tmppsi (torch.Tensor, opcional): tensor auxiliar complexo.

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação do Hamiltoniano XY.
    """

    if indice is None:
        indice = gerar_indice(L)
        
    if out is None:
        out = torch.zeros_like(psi)
    else:
        out.zeros_()

    for i in range(L):
        for j in range(i+1, L):
            if w[i, j] != 0:
                
                torch.bitwise_right_shift(indice, j - i, out=tmp)
                tmp.bitwise_xor_(indice)
                tmp.bitwise_right_shift_(i)
                tmp.bitwise_and_(1)
                flip = (1 << i) | (1 << j)
                tmp.mul_(flip)
                tmp.bitwise_xor_(indice)

                torch.index_select(psi, 0, tmp, out=tmppsi)

                out.add_(tmpsi)
                
    return out

