import torch
from .bitops import *

def X(psi : torch.Tensor, L : int, i : int, indice : torch.Tensor = None):
    """
    Aplica a porta de Pauli-X no qubit `i` do vetor de estado `psi`.

    Essa função retorna o vetor resultante da aplicação de X_i sobre `psi`, onde X_i atua como
    um operador de troca de bits, invertendo o valor do qubit `i` (|0⟩ <-> |1⟩) em cada base computacional.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do qubit sobre o qual a porta X será aplicada.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.
      Se não fornecido, será gerado automaticamente.

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação da operação X_i em `psi`.
    """

    if indice is None:
        indice = gerar_indice(L)

    novo_indice = indice ^ (1 << i)

    return psi[novo_indice]


def Z(psi : torch.Tensor, L : int, i : int, indice : torch.Tensor = None):
    """
    Aplica a porta de Pauli-Z no qubit `i` do vetor de estado `psi`.

    A operação Z_i atua multiplicando por +1 os estados com o qubit `i` igual a 0
    e por -1 os estados com o qubit `i` igual a 1. Esta função realiza essa operação
    de forma vetorial, utilizando manipulação de bits nos índices.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L ou batch x 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do qubit sobre o qual a porta Z será aplicada.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.
      Se não fornecido, será gerado automaticamente. É ajustado automaticamente se `psi` estiver em batch.

    Retorna:
    - Zipsi (torch.Tensor): vetor resultante da aplicação da operação Z_i em `psi`.
    """

    if indice is None:
        indice = gerar_indice(L)

    if psi.dim() == 2:
        indice_z = indice.unsqueeze(1)
    else:
        indice_z = indice

    Zipsi = (1 - 2 * ((indice_z >> i) & 1)) * psi

    return Zipsi


def Y(psi : torch.Tensor, L : int, i : int, indice : torch.Tensor = None):
    """
    Aplica a porta de Pauli-Y no qubit `i` do vetor de estado `psi`.

    A operação Y_i pode ser representada como Y_i = 1j * Z_i * X_i, combinando as ações
    das portas Z e X com um fator imaginário. A função aplica essas operações sequencialmente
    e multiplica o resultado por 1j para garantir a definição correta do operador Y.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do qubit sobre o qual a porta Y será aplicada.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.
      Se não fornecido, será gerado automaticamente.

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação da operação Y_i em `psi`.
    """

    if indice is None:
        indice = gerar_indice(L)

    psi = Z(psi, L, i, indice)
    psi = X(psi, L, i, indice)

    return 1j * psi


def Had(psi : torch.Tensor, L : int, i : int, indice : torch.Tensor = None):
    """
    Aplica a porta de Hadamard no qubit `i` do vetor de estado `psi`.

    A operação Hadamard é definida por:
        H = (X + Z) / √2

    Esta função aplica as portas de Pauli-X e Pauli-Z no qubit `i`, soma os resultados e normaliza
    pelo fator √2, correspondendo à definição do operador de Hadamard.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do qubit sobre o qual a porta Hadamard será aplicada.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.
      Se não fornecido, será gerado automaticamente.

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação da operação Hadamard no qubit `i`.
    """

    if indice is None:
        indice = gerar_indice(L)

    Xpi = X(psi, L, i, indice)
    Zpi = Z(psi, L, i, indice)

    return (Xpi + Zpi) / torch.sqrt(torch.tensor(2.0, device=psi.device))


def S(psi : torch.Tensor, L : int, i : int, indice : torch.Tensor = None):
    """
    Aplica a porta de fase S no qubit `i` do vetor de estado `psi`.

    A porta S é uma rotação de fase definida por:
        S = diag(1, 1j)
    Ela multiplica a componente do estado com o qubit `i` em |1⟩ por 1j, deixando a componente |0⟩ inalterada.

    A implementação verifica o valor do bit na posição `i` e aplica o fator 1j nas componentes onde o bit é 1.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do qubit sobre o qual a porta S será aplicada.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.
      Se não fornecido, será gerado automaticamente.

    Retorna:
    - Spsi (torch.Tensor): vetor resultante da aplicação da operação de fase S no qubit `i`.
    """

    if indice is None:
        indice = gerar_indice(L)

    if psi.dim() == 2:
        indice_z = indice.unsqueeze(1)
    else:
        indice_z = indice

    Spsi = (1 + ((indice_z >> i) & 1) * (1j - 1)) * psi

    return Spsi


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

    if psi.dim() == 2:
        indice_z = indice.unsqueeze(1)
    else:
        indice_z = indice

    Sdpsi = (1 + ((indice_z >> i) & 1) * (-1j - 1)) * psi

    return Sdpsi


def Rx(psi : torch.Tensor, L : int, i : int, theta : float, indice : torch.Tensor = None):
    """
    Aplica a rotação em torno do eixo X (Rx) no qubit `i` do vetor de estado `psi`.

    A operação Rx(θ) é definida por:
        Rx(θ) = exp(-i * θ * X / 2) = cos(θ/2) * I - i * sin(θ/2) * X

    A função aplica esse operador ao vetor de estado, utilizando a operação X já definida
    para trocar os estados onde o qubit `i` muda de |0⟩ para |1⟩ e vice-versa.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do qubit sobre o qual Rx será aplicado.
    - theta (float): ângulo de rotação (em radianos).
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.
      Se não fornecido, será gerado automaticamente.

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação da rotação Rx(θ) no qubit `i`.
    """

    ctheta = torch.cos(torch.tensor(theta / 2, device=psi.device))
    istheta = 1j * torch.sin(torch.tensor(theta / 2, device=psi.device))

    Xpsi = X(psi, L, i, indice)

    return ctheta * psi - istheta * Xpsi


def Ry(psi : torch.Tensor, L : int, i : int, theta : float, indice : torch.Tensor = None):
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
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.
      Se não fornecido, será gerado automaticamente.

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação da rotação Ry(θ) no qubit `i`.
    """

    ctheta = torch.cos(torch.tensor(theta / 2, device=psi.device))
    istheta = 1j * torch.sin(torch.tensor(theta / 2, device=psi.device))

    Ypsi = Y(psi, L, i, indice)

    return ctheta * psi - istheta * Ypsi


def Rz(psi : torch.Tensor, L : int, i : int, theta : float, indice : torch.Tensor = None):
    """
    Aplica a rotação em torno do eixo Z (Rz) no qubit `i` do vetor de estado `psi`.

    A operação Rz(θ) é definida por:
        Rz(θ) = exp(-i * θ * Z / 2) = cos(θ/2) * I - i * sin(θ/2) * Z

    A função aplica esse operador ao vetor de estado, utilizando a função `Z` que simula a ação
    do operador de Pauli-Z no qubit `i`.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do qubit sobre o qual Rz será aplicado.
    - theta (float): ângulo de rotação (em radianos).
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.
      Se não fornecido, será gerado automaticamente.

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação da rotação Rz(θ) no qubit `i`.
    """

    ctheta = torch.cos(torch.tensor(theta / 2, device=psi.device))
    istheta = 1j * torch.sin(torch.tensor(theta / 2, device=psi.device))

    Zpsi = Z(psi, L, i, indice)

    return ctheta * psi - istheta * Zpsi



def CNOT(psi : torch.Tensor, L : int, control : int, target : int, indice : torch.Tensor = None):
    """
    Aplica a porta CNOT ao vetor de estado `psi`, com qubit de controle `control` e qubit alvo `target`.

    A operação CNOT (Controlled-NOT) atua da seguinte forma:
        - Se o qubit de controle estiver em |1⟩, aplica uma porta X (NOT) no qubit alvo.
        - Caso contrário, não faz nada.

    A função implementa essa lógica diretamente nos índices binários dos estados base. Para cada índice `i`,
    se o bit correspondente ao qubit de controle for 1, inverte o bit correspondente ao qubit alvo.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - control (int): índice do qubit de controle.
    - target (int): índice do qubit alvo.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.
      Se não fornecido, será gerado automaticamente.

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação da porta CNOT.
    """

    if indice is None:
        indice = gerar_indice(L)

    novo_indice =  (((indice >> control) & 1) << target) ^ indice

    return psi[novo_indice]


def CZ(psi : torch.Tensor, L : int, control : int, target : int, indice : torch.Tensor = None):
    """
    Aplica a porta CZ ao vetor de estado `psi`, com qubit de controle `control` e qubit alvo `target`.

    A operação CZ (Controlled-Z) atua da seguinte forma:
        - Se ambos os qubits `control` e `target` estiverem em |1⟩, aplica um fator de fase -1.
        - Caso contrário, o estado permanece inalterado.

    A função verifica, para cada índice, se os bits correspondentes aos qubits `control` e `target` são ambos 1,
    e aplica o fator -1 nesses casos. A multiplicação é feita elemento a elemento.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - control (int): índice do qubit de controle.
    - target (int): índice do qubit alvo.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.
      Se não fornecido, será gerado automaticamente.

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação da porta CZ.
    """


    if indice is None:
        indice = gerar_indice(L)

    indice_z = ((indice >> control) & (indice >> target)) & 1
    factor = 1 - 2 * indice_z
    
    return factor.unsqueeze(1) * psi if psi.dim() == 2 else factor * psi


def SWAP(psi : torch.Tensor, L : int, i : int, j : int, indice : torch.Tensor = None):
    """
    Aplica a porta SWAP ao vetor de estado `psi`, trocando os qubits `i` e `j`.

    A operação SWAP troca os estados dos qubits `i` e `j` em todos os estados base. 

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do primeiro qubit a ser trocado.
    - j (int): índice do segundo qubit a ser trocado.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.
      Se não fornecido, será gerado automaticamente.

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação da porta SWAP.
    """


    if indice is None:
        indice = gerar_indice(L)
    
    novo_indice = permutar_bits(indice, i, j)

    return psi[novo_indice]


def XX(psi, L, i, j, indice = None):
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
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.
      Se não fornecido, será gerado automaticamente.

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação do operador XX.
    """

    if indice is None:
        indice = gerar_indice(L)

    novo_indice = indice ^ ((1 << i) | (1 << j))

    return psi[novo_indice]


def ZZ(psi, L, i, j, indice = None):
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
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.
      Se não fornecido, será gerado automaticamente.

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação do operador ZZ.
    """

    if indice is None:
        indice = gerar_indice(L)

    if psi.dim() == 2:
        indice_z = indice.unsqueeze(1)
    else:
        indice_z = indice

    novo_indice = 1 - 2 * (((indice_z >> i) ^ (indice_z >> j)) & 1)

    return psi * novo_indice


def YY(psi, L, i, j, indice = None):
    """
    Aplica o operador YY = Y_i Y_j ao vetor de estado `psi`.

    A operação é feita pela aplicação sequencial dos operadores ZZ e XX, seguidos da multiplicação por -1,
    em conformidade com a identidade Y_i Y_j = - Z_i Z_j X_i X_j.

    Parâmetros:
    - psi (torch.Tensor): vetor de estado (dimensão 2^L), representado como tensor complexo.
    - L (int): número de qubits.
    - i (int): índice do primeiro qubit.
    - j (int): índice do segundo qubit.
    - indice (torch.Tensor, opcional): tensor com os índices inteiros correspondentes aos estados base.
    Se não fornecido, será gerado automaticamente.

    Retorna:
    - (torch.Tensor): vetor resultante da aplicação do operador YY.
    """

    if indice is None:
        indice = gerar_indice(L)

    psi = ZZ(psi, L, i, j, indice)
    psi = XX(psi, L, i, j, indice)

    return - psi
