import torch

from .config import device


def gerar_indice(L):
    """
    Gera um tensor com os índices dos estados base de uma cadeia de L qubits.

    Parâmetros:
    - L (int): número de qubits.

    Retorna:
    - torch.Tensor: tensor de inteiros de 0 até 2^L - 1, no device padrão.
    """

    indice = torch.arange(2**L, dtype=torch.int32 if L <= 31 else torch.int64, device=device)
    return indice


def ligar_bit(num, p: int, out=None):
    """
    Define o bit na posição p como 1 no número num.

    Parâmetros:
    - num (int ou torch.Tensor): número inteiro ou tensor de inteiros.
    - p (int): posição do bit a ser ativada.
    - out (opcional): tensor de saída (somente se num for tensor).

    Retorna:
    - int: se num for int, retorna um novo int.
    - torch.Tensor: se num for tensor, retorna out (modificado in-place) ou um novo tensor.
    """
    valor = (1 << p)

    if isinstance(num, int):
        return num | valor

    if out is None:
        return num | valor
    else:
        torch.bitwise_or(num, valor, out=out)
        return out


def desligar_bit(num, p: int, out=None):
    """
    Define o bit na posição p como 0 no número num.

    Parâmetros:
    - num (int ou torch.Tensor): valor de entrada.
    - p (int): posição do bit a ser desativada.
    - out (opcional): tensor de saída (se num for tensor).

    Retorna:
    - int: se num for int.
    - torch.Tensor: se num for tensor.
    """
    valor = ~(1 << p)

    if isinstance(num, int):
        return num & valor

    if out is None:
        return num & valor 
    else:
        torch.bitwise_and(num, valor, out=out)
        return out



def bit(num, i: int, out=None):
    """
    Retorna o valor do bit na posição i do número num.

    Parâmetros:
    - num (int ou torch.Tensor): valor de entrada.
    - i (int): posição do bit a ser lida.
    - out (opcional): tensor de saída (se num for tensor).

    Retorna:
    - int: se num for int.
    - torch.Tensor: se num for tensor.
    """
    if isinstance(num, int):
        return (num >> i) & 1

    if out is None:
        return (num >> i) & 1
    else:
        torch.bitwise_right_shift(num, i, out=out)
        out.bitwise_and_(1)
        return out


def contar_bits(num, L: int, out=None):
    """
    Conta o número de bits iguais a 1 nos L bits menos significativos de num.

    Parâmetros:
    - num (int ou torch.Tensor): número(s) de entrada.
    - L (int): número de bits a considerar (menos significativos).
    - out (opcional): tensor de saída, onde será armazenado o resultado (se num for tensor).

    Retorna:
    - int: se num for int.
    - torch.Tensor: se num for tensor.
    """
    if isinstance(num, int):
        uns = 0
        for i in range(L):
            uns += (num >> i) & 1
        return uns

    if out is None:
        out = torch.zeros_like(num)
    else:
        out.zero_()

    tmp = torch.empty_like(num)
    for i in range(L):
        torch.bitwise_right_shift(num, i, out=tmp)
        tmp.bitwise_and_(1)   
        out.add_(tmp)

    del tmp
    return out



def permutar_bits(num, i: int, j: int, out=None):
    """
    Troca os bits nas posições i e j do número num.

    Parâmetros:
    - num (int ou torch.Tensor): número de entrada.
    - i (int): índice do primeiro bit.
    - j (int): índice do segundo bit.
    - out (opcional): tensor de saída (se num for tensor).

    Retorna:
    - int: se num for int.
    - torch.Tensor: se num for tensor.
    """
    if isinstance(num, int):
        mask = ((num >> i) ^ (num >> j)) & 1
        return num ^ ((mask << i) | (mask << j))


    if out is None:
        out = torch.empty_like(num)

    tmp = num >> j
    
    torch.bitwise_right_shift(num, i, out=out)

    tmp.bitwise_xor_(out)
    tmp.bitwise_and_(1) # mask

    torch.bitwise_left_shift(tmp, j, out=out) # mask << j
    tmp.bitwise_left_shift_(i) # mask << i
    tmp.bitwise_or_(out) # (mask << i) | (mask << j)
    torch.bitwise_xor(num, tmp, out=out)

    del tmp
    return out


def possui_um_bit_1(num, out=None):
    """
    Verifica se o número possui exatamente um bit 1.

    Parâmetros:
    - num (int ou torch.Tensor): número de entrada.
    - out (opcional): tensor de saída booleano (se num for tensor).

    Retorna:
    - bool: se num for int.
    - torch.Tensor (bool): se num for tensor.
    """
    if isinstance(num, int):
        return (num != 0) and ((num & (num - 1)) == 0)

    if out is None:
        out = torch.empty_like(num, dtype=torch.bool)

    tmp = num - 1
    tmp.bitwise_and_(num)
    out.copy_((num != 0) & (tmp == 0))
    
    del tmp
    return out


def translacao(num, d: int, L: int, out=None):
    """
    Aplica uma translação cíclica de d posições para a direita nos L bits de num.

    Parâmetros:
    - num (int ou torch.Tensor): número de entrada.
    - d (int): número de posições a deslocar.
    - L (int): número total de bits considerados.
    - out (opcional): tensor de saída (se num for tensor).

    Retorna:
    - int: se num for int.
    - torch.Tensor: se num for tensor.
    """
    d %= L
    mask = (1 << L) - 1

    if isinstance(num, int):
        return (num >> (L - d)) | ((num << d) & mask)


    if out is None:
        out = torch.empty_like(num)

    tmp = num << d

    torch.bitwise_right_shift(num, L - d, out=out)
    tmp.bitwise_and_(mask)
    out.bitwise_or_(tmp)
    
    del tmp
    return out


def inversao(num, L: int, out=None):
    """
    Inverte todos os bits nos L bits menos significativos de num.

    Parâmetros:
    - num (int ou torch.Tensor): número de entrada.
    - L (int): número de bits considerados.
    - out (opcional): tensor de saída (se num for tensor).

    Retorna:
    - int: se num for int.
    - torch.Tensor: se num for tensor.
    """
    mask = (1 << L) - 1

    if isinstance(num, int):
        return (~num) & mask

    if out is None:
        out = torch.empty_like(num)

    torch.bitwise_not(num, out=out)
    out.bitwise_and_(mask)
    return out


def reflexao(num, L: int, out=None):
    """
    Reflete a ordem dos L bits de num (espelhamento da bitstring).

    Parâmetros:
    - num (int ou torch.Tensor): número de entrada.
    - L (int): número total de bits.
    - out (opcional): tensor de saída (se num for tensor).

    Retorna:
    - int: se num for int.
    - torch.Tensor: se num for tensor.
    """
    if isinstance(num, int):
        n1 = 0
        for i in range(L):
            n1 |= ((num >> i) & 1) << (L - 1 - i)
        return n1


    if out is None:
        out = torch.zeros_like(num)
    else:
        out.zero_()

    tmp = torch.empty_like(num)
    for i in range(L):
        torch.bitwise_right_shift(num, i, out=tmp)
        tmp.bitwise_and_(1)             # tmp = (num >> i) & 1
        tmp.bitwise_left_shift_(L-1-i)  # tmp = tmp << (L - 1 - i)          
        out.bitwise_or_(tmp)                      # soma os bits espelhados

    return out

