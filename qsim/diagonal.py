import torch
from .config import device, dtype


def operador_diagonal(L: int, op: torch.Tensor, pos: list[int], coef: complex = 1):
    """
    Gera a diagonal principal de um operador quântico definido como o produto tensorial de operadores locais 
    com representação matricial diagonal (como Z ou identidade), atuando sobre uma cadeia de L qubits.

    Parâmetros:
    - L (int): número total de qubits da cadeia.
    - op (torch.Tensor): tensor com 2 elementos representando a diagonal do operador de 1 qubit 
                         (exemplo: Z = [1, -1], identidade = [1, 1]).
    - pos (list[int]): lista com as posições dos qubits onde o operador deve ser aplicado.
    - coef (complex, opcional): coeficiente multiplicativo aplicado ao resultado (default = 1).

    Retorna:
    - torch.Tensor: vetor com 2^L elementos correspondentes à diagonal do operador total.
    """

    if coef == 0:
        return torch.zeros(2**L, dtype=dtype, device=device)

    pos = sorted(pos)
    resultado = torch.tensor([coef], dtype=dtype, device=device)

    j = 0
    for i in pos:
        if i > j:
            ident = torch.ones(2**(i - j), dtype=dtype, device=device)
            resultado = torch.kron(resultado, ident)
        resultado = torch.kron(resultado, op)
        j = i + 1

    if j < L:
        ident = torch.ones(2**(L - j), dtype=dtype, device=device)
        resultado = torch.kron(resultado, ident)

    return resultado


def cadeia_z(L: int, pos: list[int], coef: complex = 1):
    """
    Gera a diagonal principal de um operador do tipo Z_i Z_j ... em uma cadeia de L qubits.

    Parâmetros:
    - L (int): número total de qubits.
    - pos (list[int]): lista com os índices dos qubits nos quais o operador Z atua.
    - coef (complex, opcional): fator multiplicativo do operador. Default é 1.

    Retorna:
    - torch.Tensor: vetor com 2^L elementos representando a diagonal principal do operador resultante.
    """
    op_z = torch.tensor([1.0, -1.0], dtype=dtype, device=device)
    return operador_diagonal(L, op_z, pos, coef)


def cadeia_numero(L: int, pos: list[int], coef: complex = 1):
    """
    Gera a diagonal principal do operador do tipo n_i n_j ... em uma cadeia de L qubits.

    O operador número, n_i, é definido como n_i = (1 - Z_i) / 2, valendo 0 para o estado |0⟩ e 1 para o estado |1⟩.

    Parâmetros:
    - L (int): número total de qubits.
    - pos (list[int]): lista com os índices dos qubits nos quais o operador número atua.
    - coef (complex, opcional): fator multiplicativo do operador. Default é 1.

    Retorna:
    - torch.Tensor: vetor com 2^L elementos representando a diagonal principal do operador resultante.
    """
    op_n = torch.tensor([0.0, 1.0], dtype=dtype, device=device)  # (1 - Z)/2 = [0, 1]
    return operador_diagonal(L, op_n, pos, coef)
