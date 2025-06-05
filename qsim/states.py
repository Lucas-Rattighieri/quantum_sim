import torch

from .config import device, dtype


#@title Estados iniciais

def superposicao_uniforme(L, num_estados = 1):

    if num_estados == 1:
        psi = torch.ones(2**L, dtype=dtype, device=device)
    else:
        psi = torch.ones(2 ** L, num_estados, dtype=dtype, device=device)

    psi /= torch.sqrt(torch.tensor(2 ** L, device=device))

    return psi


def base_z(L):
    return torch.eye(2**L, dtype=dtype, device=device)


def base_x(L):
    base = torch.arange(2**L, dtype=torch.int64, device=device).unsqueeze(1)  # shape: (2**L, 1)

    matriz = base.repeat(1, 2**L)

    produto = 1 - 2 * (contar_bits(matriz.T & matriz, L) & 1)

    psi = produto / torch.sqrt(torch.tensor(2 ** L, device=device))

    return psi



def autoestado_z(L, i):
    psi = torch.zeros(2**L, dtype=dtype, device=device)
    psi[i] = 1
    return psi


def autoestado_x(L, i):
    indice = torch.arange(2**L, dtype=torch.int64, device=device)

    novo_indice = indice & i
    produto = 1 - 2 * (contar_bits(novo_indice, L) & 1)

    psi = produto / torch.sqrt(torch.tensor(2 ** L, device=device))

    return psi
