import torch
from .bitops import *

def X(psi: torch.Tensor, L: int, i: int, indice: torch.Tensor = None, tmp: torch.Tensor = None, out: torch.Tensor = None):
    """
    Aplica a porta de Pauli-X no qubit `i` do vetor de estado `psi`.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do qubit sobre o qual a porta X será aplicada.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros dos estados base.
    - tmp (torch.Tensor, opcional): tensor auxiliar para armazenar os índices modificados.
    - out (torch.Tensor, opcional): vetor de saída para o resultado (evita alocação).

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação da operação X_i em `psi`.
    """

    if indice is None:
        indice = gerar_indice(L)

    if novo_indice is None:
        tmp = indice ^ (1 << i)
    else:
        torch.bitwise_xor(indice, 1 << i, out=tmp)

    if out is None:
        return psi[tmp]
    else:
        torch.index_select(psi, dim, tmp, out=out)
        return out


def Z(psi: torch.Tensor, L: int, i: int, indice: torch.Tensor = None, tmp: torch.Tensor = None, out: torch.Tensor = None):
    """
    Aplica a porta de Pauli-Z no qubit `i` do vetor de estado `psi`.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L ou batch x 2^L), tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do qubit sobre o qual a porta Z será aplicada.
    - indice (torch.Tensor, opcional): índices dos estados base (gerados se não fornecido).
    - tmp (torch.Tensor, opcional): tensor auxiliar inteiro para armazenar os bits (evita alocação).
    - out (torch.Tensor, opcional): vetor de saída para o resultado (evita alocação).

    Retorna:
    - torch.Tensor: resultado da aplicação da operação Z_i em `psi`.
    """

    if indice is None:
        indice = gerar_indice(L)

    if tmp is None:
        tmp = bit(indice, i)
    else:
        bit(indice, i, tmp)

    tmp.mul_(-2).add_(1)

    if psi.dim() == 2:
        tmpl = tmp.unsqueeze(1)
    else:
        tmpl = tmp

    if out is None:
        return tmpl * psi
    else:
        torch.mul_(tmpl, psi, out=out)
        return out



def Y(psi: torch.Tensor, L: int, i: int, indice: torch.Tensor = None, tmp: torch.Tensor = None, out: torch.Tensor = None):
    """
    Aplica a porta de Pauli-Y no qubit `i` do vetor de estado `psi`.

    A operação Y_i é definida como Y_i = i * Z_i * X_i.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (2^L,) ou (2^L, N), tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do qubit sobre o qual a porta Y será aplicada.
    - indice (torch.Tensor, opcional): tensor com os índices dos estados base.
    - tmp (torch.Tensor, opcional): tensor auxiliar para evitar alocações.
    - out (torch.Tensor, opcional): vetor de saída para o resultado (evita alocação).

    Retorna:
    - torch.Tensor: vetor resultante da aplicação da operação Y_i.
    """

    if indice is None:
        indice = gerar_indice(L)



    out = X(psi, L, i, indice, tmp=tmp, out=out)

    out = Z(out, L, i, indice, tmp=tmp, out=out)
    
    out.mul_(-1j)  # multiplicação in-place

    return out


def Had(psi: torch.Tensor, L: int, i: int, indice: torch.Tensor = None, tmp: torch.Tensor = None):
    """
    Aplica a porta de Hadamard no qubit `i` do vetor de estado `psi`.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L ou 2^L × N), tensor complexo.
    - L (int): número de qubits.
    - i (int): qubit sobre o qual H será aplicado.
    - indice (torch.Tensor, opcional): índices dos estados base.
    - tmp (torch.Tensor, opcional): auxiliar inteiro para uso interno da porta Z.

    Retorna:
    - torch.Tensor: vetor resultante da aplicação de Hadamard.
    """

    if indice is None:
        indice = gerar_indice(L)

    Zpi = Z(psi, L, i, indice, tmp=tmp)
    Xpi = X(psi, L, i, indice, tmp=tmp)
    Zpi.add_(Xpi)
    Zpi.div_(torch.sqrt(torch.tensor(2.0, device=psi.device)))

    return Zpi



def S(psi: torch.Tensor, L: int, i: int, indice: torch.Tensor = None, tmp: torch.Tensor = None, out: torch.Tensor = None):
    """
    Aplica a porta de fase S no qubit `i` do vetor de estado `psi`.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L ou 2^L x N), tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do qubit sobre o qual a porta S será aplicada.
    - indice (torch.Tensor, opcional): índices dos estados base.
    - tmp (torch.Tensor, opcional): auxiliar inteiro para armazenar os bits (evita alocação).
    - out (torch.Tensor, opcional): vetor de saída para resultado (in-place se fornecido).

    Retorna:
    - torch.Tensor: vetor resultante da aplicação de S_i.
    """

    if indice is None:
        indice = gerar_indice(L)

    if tmp is None:
        tmp = bit(indice, i)
    else:
        bit(indice, i, tmp)

    tmp.mul_(1j - 1).add_(1)

    if psi.dim() == 2:
        factor = tmp.unsqueeze(1)
    else:
        factor = tmp

    if out is None:
        return fator * psi
    else:
        torch.mul_(factor, psi, out=out)
        return out



def Sd(psi : torch.Tensor, L : int, i : int, indice : torch.Tensor = None):
    """
    Aplica a porta de fase conjugada S† (S dagger) no qubit `i` do vetor de estado `psi`.

    A porta S† é a inversa da porta de fase S e é definida por:
        S† = diag(1, -1j)
    Ela multiplica a componente do estado com o qubit `i` em |1⟩ por -1j, deixando a componente |0⟩ inalterada.

    A implementação verifica o valor do bit na posição `i` e aplica o fator -1j nas componentes onde o bit é 1.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do qubit sobre o qual a porta S† será aplicada.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.
      Se não fornecido, será gerado automaticamente.

    Retorna:
    - Sdpsi (torch.Tensor): vetor resultante da aplicação da operação de fase S† no qubit `i`.
    """

    if indice is None:
        indice = gerar_indice(L)

    if tmp is None:
        tmp = bit(indice, i)
    else:
        bit(indice, i, tmp)

    tmp.mul_(-1j - 1).add_(1)

    if psi.dim() == 2:
        factor = tmp.unsqueeze(1)
    else:
        factor = tmp

    if out is None:
        return fator * psi
    else:
        torch.mul_(factor, psi, out=out)
        return out


def Rx(psi: torch.Tensor, L: int, i: int, theta: float, indice: torch.Tensor = None, tmp: torch.Tensor = None):
    """
    Aplica a rotação Rx(θ) no qubit `i` do vetor de estado `psi`.

    Rx(θ) = exp(-i * θ * X / 2) = cos(θ/2) * I - i * sin(θ/2) * X

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (2^L ou 2^L x N), tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do qubit sobre o qual Rx será aplicado.
    - theta (float): ângulo de rotação em radianos.
    - indice (torch.Tensor, opcional): índices dos estados base.
    - tmp (torch.Tensor, opcional): vetor auxiliar de índices para X.

    Retorna:
    - torch.Tensor: resultado de Rx(θ) aplicado em `psi`.
    """

    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=psi.dtype, device=psi.device)
    else:
        theta = theta.to(dtype=psi.dtype, device=psi.device)

    ctheta = torch.cos(theta / 2)
    istheta = 1j * torch.sin(theta / 2)

    Xpsi = X(psi, L, i, indice, tmp=tmp)
    Xpsi.mul_(-istheta)
    Xpsi.add_(psi, alpha=ctheta)

    return Xpsi



def Ry(psi : torch.Tensor, L : int, i : int, theta : float, indice : torch.Tensor = None, tmp: torch.Tensor = None):
    """
    Aplica a rotação em torno do eixo Y (Ry) no qubit `i` do vetor de estado `psi`.

    A operação Ry(θ) é definida por:
        Ry(θ) = exp(-i * θ * Y / 2) = cos(θ/2) * I - i * sin(θ/2) * Y

    A função aplica esse operador ao vetor de estado, utilizando a função `Y` que simula a ação
    do operador de Pauli-Y no qubit `i`.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do qubit sobre o qual Ry será aplicado.
    - theta (float): ângulo de rotação (em radianos).
    - indice (torch.Tensor, opcional): índices dos estados base.
    - tmp (torch.Tensor, opcional): vetor auxiliar de índices para Y.

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação da rotação Ry(θ) no qubit `i`.
    """

    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=psi.dtype, device=psi.device)
    else:
        theta = theta.to(dtype=psi.dtype, device=psi.device)

    ctheta = torch.cos(theta / 2)
    istheta = 1j * torch.sin(theta / 2)

    Ypsi = Y(psi, L, i, indice, tmp)
    Ypsi.mul_(-istheta)
    Ypsi.add_(psi, alpha=ctheta)

    return Ypsi


def Rz(psi: torch.Tensor, L: int, i: int, theta: float, indice: torch.Tensor = None, tmp: torch.Tensor = None, out: torch.Tensor = None):
    """
    Aplica a rotação Rz(θ) no qubit `i` do vetor de estado `psi`.

    Rz(θ) = exp(-i * θ * Z / 2) = cos(θ/2) * I - i * sin(θ/2) * Z

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (2^L ou 2^L x N), tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do qubit sobre o qual Rz será aplicado.
    - theta (float): ângulo de rotação em radianos.
    - indice (torch.Tensor, opcional): índices dos estados base.
    - tmp (torch.Tensor, opcional): vetor auxiliar de inteiros para uso em Z.
    - out (torch.Tensor, opcional): vetor de saída onde será armazenado o resultado.

    Retorna:
    - torch.Tensor: resultado de Rz(θ) aplicado a `psi`.
    """

    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=psi.dtype, device=psi.device)
    else:
        theta = theta.to(dtype=psi.dtype, device=psi.device)

    ctheta = torch.cos(theta / 2)
    istheta = 1j * torch.sin(theta / 2)
    

    Zpsi = Z(psi, L, i, indice, tmp=tmp, out=out)

    Zpsi = Y(psi, L, i, indice, tmp)
    Zpsi.mul_(-istheta)
    Zpsi.add_(psi, alpha=ctheta)

    return Zpsi
    



def CNOT(psi: torch.Tensor, L: int, control: int, target: int, indice: torch.Tensor = None, tmp: torch.Tensor = None):
    """
    Aplica a porta CNOT ao vetor de estado `psi`, com qubit de controle `control` e qubit alvo `target`.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (2^L ou 2^L x N), tensor complexo.
    - L (int): número de qubits.
    - control (int): qubit de controle.
    - target (int): qubit alvo.
    - indice (torch.Tensor, opcional): índices dos estados base.
    - tmp (torch.Tensor, opcional): tensor auxiliar para armazenar os índices modificados.

    Retorna:
    - torch.Tensor: vetor resultante da aplicação da CNOT.
    """

    if indice is None:
        indice = gerar_indice(L)

    if tmp is None:
        tmp = bit(indice, control)
    else:
        bit(indice, control, tmp)

    tmp.bitwise_left_shift_(target)
    tmp.bitwise_xor_(indice)
        
    return psi[tmp]



def CZ(psi: torch.Tensor, L: int, control: int, target: int,
       indice: torch.Tensor = None, tmp: torch.Tensor = None, out: torch.Tensor = None):
    """
    Aplica a porta CZ ao vetor de estado `psi`, com qubit de controle `control` e qubit alvo `target`.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (2^L ou 2^L x N), tensor complexo.
    - L (int): número de qubits.
    - control (int): índice do qubit de controle.
    - target (int): índice do qubit alvo.
    - indice (torch.Tensor, opcional): índices dos estados base.
    - tmp (torch.Tensor, opcional): tensor auxiliar (mesma forma de `indice`).
    - out (torch.Tensor, opcional): vetor de saída (se fornecido, operação é in-place).

    Retorna:
    - torch.Tensor: resultado da aplicação de CZ.
    """

    if indice is None:
        indice = gerar_indice(L)

    if target < control:
        control, target = target, control
           
    if tmp is None:
        tmp = indice.clone()
    else:
        tmp.copy_(indice)

    tmp.bitwise_right_shift_(target - control)
    tmp.bitwise_and_(indice)
    tmp.bitwise_right_shift_(control)
    tmp.bitwise_and_(1)
           
    tmp.mul_(-2).add_(1)

    if psi.dim() == 2:
        factor = tmp.unsqueeze(1)
    else:
        factor = tmp

    if out is None:
        return factor * psi
    else:
        torch.mul(psi, factor, out=out)
        return out



def SWAP(psi : torch.Tensor, L : int, i : int, j : int, indice : torch.Tensor = None, tmp: torch.Tensor = None):
    """
    Aplica a porta SWAP ao vetor de estado `psi`, trocando os qubits `i` e `j`.

    A operação SWAP troca os estados dos qubits `i` e `j` em todos os estados base. 

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do primeiro qubit a ser trocado.
    - j (int): índice do segundo qubit a ser trocado.
    - indice (torch.Tensor, opcional): índices dos estados base.
    - tmp (torch.Tensor, opcional): tensor auxiliar (mesma forma de `indice`).

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação da porta SWAP.
    """


    if indice is None:
        indice = gerar_indice(L)
    
    tmp = permutar_bits(indice, i, j, tmp)

    return psi[tmp]


def XX(psi, L, i, j, indice : torch.Tensor = None, tmp: torch.Tensor = None):
    """
    Aplica o operador XX = X_i X_j ao vetor de estado `psi`.

    A função executa a ação da combinação de portas de Pauli-X nos qubits `i` e `j`,
    o que equivale a inverter simultaneamente os bits nas posições `i` e `j` para cada
    estado da base computacional. Essa operação implementa a troca de população entre
    os estados |01⟩ e |10⟩ dos qubits `i` e `j`.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do primeiro qubit.
    - j (int): índice do segundo qubit.
    - indice (torch.Tensor, opcional): índices dos estados base.
    - tmp (torch.Tensor, opcional): tensor auxiliar (mesma forma de `indice`).

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação do operador XX.
    """

    if indice is None:
        indice = gerar_indice(L)

    if tmp is None:
        tmp = indice ^ ((1 << i) | (1 << j))
    else:
        torch.bitwise_xor(indice, (1 << i) | (1 << j), out=tmp)

    return psi[tmp]


def ZZ(psi, L, i, j, indice = None, tmp: torch.Tensor = None, out: torch.Tensor = None):
    """
    Aplica o operador ZZ = Z_i Z_j ao vetor de estado `psi`.

    A função multiplica cada componente do vetor `psi` pelo autovalor correspondente 
    do operador Z_i Z_j, que é +1 se os bits nos qubits `i` e `j` forem iguais, e -1 se forem diferentes. 
    Essa operação atua diagonalmente na base computacional, alterando apenas os sinais dos coeficientes.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do primeiro qubit.
    - j (int): índice do segundo qubit.
    - indice (torch.Tensor, opcional): índices dos estados base.
    - tmp (torch.Tensor, opcional): tensor auxiliar (mesma forma de `indice`).
    - out (torch.Tensor, opcional): tensor para resultado in-place.

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação do operador ZZ.
    """

    if indice is None:
        indice = gerar_indice(L)

   
    if j < i:
        i, j = j, i
           
    if tmp is None:
        tmp = indice.clone()
    else:
        tmp.copy_(indice)

    tmp.bitwise_right_shift_(j - i)
    tmp.bitwise_xor_(indice)
    tmp.bitwise_right_shift_(i)
    tmp.bitwise_and_(1)
           
    tmp.mul_(-2).add_(1)

    if psi.dim() == 2:
        factor = tmp.unsqueeze(1)
    else:
        factor = tmp

    if out is None:
        return factor * psi
    else:
        torch.mul(psi, factor, out=out)
        return out


def YY(psi, L, i, j, indice = None, tmp: torch.Tensor = None):
    """
    Aplica o operador YY = Y_i Y_j ao vetor de estado `psi`.

    A operação é feita pela aplicação sequencial dos operadores ZZ e XX, seguidos da multiplicação por -1,
    em conformidade com a identidade Y_i Y_j = - Z_i Z_j X_i X_j.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do primeiro qubit.
    - j (int): índice do segundo qubit.
    - indice (torch.Tensor, opcional): índices dos estados base.
    - tmp (torch.Tensor, opcional): tensor auxiliar (mesma forma de `indice`).

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação do operador YY.
    """

    if indice is None:
        indice = gerar_indice(L)

    psi = ZZ(psi, L, i, j, indice, tmp)
    psi = XX(psi, L, i, j, indice, tmp)
    psi.mul_(-1)

    return psi
