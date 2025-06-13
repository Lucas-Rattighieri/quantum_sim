import torch
from .operators import Z, X, Y, ZZ, XX, YY


def expHx(psi : torch.Tensor, L : int, theta : float, indice : torch.Tensor = None):
    """
    Aplica a evolução temporal associada ao termo do Hamiltoniano do tipo Σ_{a} X_a 
    sobre o vetor de estado `psi`.

    A operação corresponde à aplicação sucessiva de exp(-i θ X_a) para cada qubit `a` no sistema, 
    o que equivale à evolução gerada por um termo de campo magnético transversal (eixo X) no Hamiltoniano.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - theta (float): ângulo de rotação aplicado a cada X_a.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.

    Retorna:
    - psi (torch.Tensor): vetor de estado após a aplicação da evolução X.
    """

    if indice is None:
        indice = gerar_indice(L)

    ctheta = torch.cos(torch.tensor(theta, device=psi.device))
    istheta = 1j * torch.sin(torch.tensor(theta, device=psi.device))

    for a in range(L):
        cpsi = psi * ctheta
        isXpsi = X(psi, L, a, indice) * istheta
        psi = cpsi - isXpsi

    return psi


def expHy(psi : torch.Tensor, L : int, theta : float, indice : torch.Tensor = None):
    """
    Aplica a evolução temporal associada ao termo do Hamiltoniano do tipo Σ_{a} Y_a 
    sobre o vetor de estado `psi`.

    A operação corresponde à aplicação sucessiva de exp(-i θ Y_a) para cada qubit `a` no sistema, 
    o que equivale à evolução gerada por um termo de campo magnético transversal (eixo Y) no Hamiltoniano.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - theta (float): ângulo de rotação aplicado a cada Y_a.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.

    Retorna:
    - psi (torch.Tensor): vetor de estado após a aplicação da evolução Y.
    """

    if indice is None:
        indice = gerar_indice(L)
        
    ctheta = torch.cos(torch.tensor(theta, device=psi.device))
    istheta = 1j * torch.sin(torch.tensor(theta, device=psi.device))

    for a in range(L):
        cpsi = psi * ctheta
        isXpsi = Y(psi, L, a, indice) * istheta
        psi = cpsi - isXpsi

    return psi


def expHz(psi : torch.Tensor, L : int, theta : float, indice : torch.Tensor = None):
    """
    Aplica a evolução temporal associada ao termo do Hamiltoniano do tipo Σ_{a} Z_a 
    sobre o vetor de estado `psi`.

    A operação corresponde à aplicação sucessiva de exp(-i θ Z_a) para cada qubit `a` no sistema, 
    o que equivale à evolução gerada por um termo de campo magnético longitudinal no Hamiltoniano.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - theta (float): ângulo de rotação aplicado a cada Z_a.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.

    Retorna:
    - psi (torch.Tensor): vetor de estado após a aplicação da evolução Z.
    """

    if indice is None:
        indice = gerar_indice(L)
        
    ctheta = torch.cos(torch.tensor(theta, device=psi.device))
    istheta = 1j * torch.sin(torch.tensor(theta, device=psi.device))

    for a in range(L):
        cpsi = psi * ctheta
        isXpsi = Z(psi, L, a, indice) * istheta
        psi = cpsi - isXpsi

    return psi


def expHxx(psi : torch.Tensor, L : int, w, theta : float, indice : torch.Tensor = None):
    """
    Aplica a evolução temporal associada ao termo do Hamiltoniano do tipo Σ_{i<j} X_i X_j, w_{i,j} != 0 
    sobre o vetor de estado `psi`.

    A operação realizada é equivalente à aplicação de exp(-i θ X_i X_j) para cada par (i < j) 
    com acoplamento não-nulo w[i, j], agindo sobre o vetor de estado global.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - w (array ou tensor): matriz simétrica indicando os acoplamentos. w[i, j] deve ser 1 (ativa interação) ou 0 (sem interação).
    - theta (float): ângulo de rotação.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.

    Retorna:
    - psi (torch.Tensor): vetor de estado após a aplicação da evolução X_i X_j.
    """

    if indice is None:
        indice = gerar_indice(L)
        
    ctheta = torch.cos(torch.tensor(theta, device=psi.device))
    istheta = 1j * torch.sin(torch.tensor(theta, device=psi.device))

    for i in range(L):
        for j in range(i + 1, L):
            if w[i, j] != 0:
                cpsi = psi * ctheta
                isXXpsi = XX(psi, L, i, j, indice) * istheta
                psi = cpsi - isXXpsi

    return psi


def expHyy(psi : torch.Tensor, L : int, w, theta : float, indice : torch.Tensor = None):
    """
    Aplica a evolução temporal associada ao termo do Hamiltoniano do tipo Σ_{i<j} Y_i Y_j, w_{i,j} != 0 
    sobre o vetor de estado `psi`.

    A operação realizada é equivalente à aplicação de exp(-i θ Y_i Y_j) para cada par (i < j) 
    com acoplamento não-nulo w[i, j], agindo sobre o vetor de estado global.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - w (array ou tensor): matriz simétrica indicando os acoplamentos. w[i, j] deve ser 1 (ativa interação) ou 0 (sem interação).
    - theta (float): ângulo de rotação.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.

    Retorna:
    - psi (torch.Tensor): vetor de estado após a aplicação da evolução Y_i Y_j.
    """

    if indice is None:
        indice = gerar_indice(L)
        
    ctheta = torch.cos(torch.tensor(theta, device=psi.device))
    istheta = 1j * torch.sin(torch.tensor(theta, device=psi.device))

    for i in range(L):
        for j in range(i + 1, L):
            if w[i, j] != 0:
                cpsi = psi * ctheta
                isYYpsi = YY(psi, L, i, j, indice) * istheta
                psi = cpsi - isYYpsi

    return psi


def expHzz(psi : torch.Tensor, L : int, w, theta : float, indice : torch.Tensor = None):
    """
    Aplica a evolução temporal associada ao termo do Hamiltoniano do tipo Σ_{i<j} Z_i Z_j, w_{i,j} != 0 
    sobre o vetor de estado `psi`.

    A operação realizada é equivalente à aplicação de exp(-i θ Z_i Z_j) para cada par (i < j) 
    com acoplamento não-nulo w[i, j], agindo sobre o vetor de estado global.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - w (array ou tensor): matriz simétrica indicando os acoplamentos. w[i, j] deve ser 1 (ativa interação) ou 0 (sem interação).
    - theta (float): ângulo de rotação.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.

    Retorna:
    - psi (torch.Tensor): vetor de estado após a aplicação da evolução Z_i Z_j.
    """

    if indice is None:
        indice = gerar_indice(L)
        
    ctheta = torch.cos(torch.tensor(theta, device=psi.device))
    istheta = 1j * torch.sin(torch.tensor(theta, device=psi.device))

    for i in range(L):
        for j in range(i + 1, L):
            if w[i, j] != 0:
                cpsi = psi * ctheta
                isYYpsi = ZZ(psi, L, i, j, indice) * istheta
                psi = cpsi - isYYpsi

    return psi


def expHxy(psi : torch.Tensor, L : int, w, theta : float, indice : torch.Tensor = None):
    """
    Aplica a evolução unária gerada pelo Hamiltoniano Σ_{i<j} (X_i X_j + Y_i Y_j) / 2, w_{i,j} != 0 em um vetor de estado `psi`.

    A evolução corresponde à aplicação de operadores do tipo exp(-i θ (X_i X_j + Y_i Y_j)) 
    para todos os pares (i < j) com acoplamento não-nulo w[i, j].

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - w (array ou tensor): matriz simétrica indicando os acoplamentos. w[i, j] deve ser 1 (ativa interação) ou 0 (sem interação).
    - theta (float): ângulo de rotação associado à evolução.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base (de 0 até 2^L - 1).

    A função aplica rotações no subespaço formado pelos pares de estados |01⟩ e |10⟩, 
    preservando |00⟩ e |11⟩. 

    Retorna:
    - epsi (torch.Tensor): novo vetor de estado após aplicação da evolução XY.
    """

    if indice is None:
        indice = gerar_indice(L)
        
    epsi = psi.clone()

    ctheta = torch.cos(torch.tensor(theta, device=psi.device))
    istheta = 1j * torch.sin(torch.tensor(theta, device=psi.device))

    for i in range(L):
        for j in range(i + 1, L):
            if w[i, j] != 0:

                mask01 = (((~indice >> i) & (indice >> j)) & 1) == 1
                flip = (1 << i) | (1 << j)

                indices01 = indice[mask01]
                indices10 = indices01 ^ flip
 
                psi01 = epsi[indices01].clone()
                psi10 = epsi[indices10].clone()

                epsi[indices01] = ctheta * psi01 - istheta * psi10
                epsi[indices10] = ctheta * psi10 - istheta * psi01

    return epsi
