import torch

def X(psi, L, i, indice = None):

    if indice is None:
        indice = torch.arange(2**L, dtype=torch.int64, device=psi.device)

    novo_indice = indice ^ (1 << i)

    return psi[novo_indice]


def Z(psi, L, i, indice = None):

    if indice is None:
        indice = torch.arange(2**L, dtype=torch.int64, device=psi.device)

    if psi.dim() == 2:
        indice_z = indice.unsqueeze(1)
    else:
        indice_z = indice

    Zipsi = (1 - 2 * ((indice_z >> i) & 1)) * psi

    return Zipsi


def Y(psi, L, i, indice = None):

    if indice is None:
        indice = torch.arange(2**L, dtype=torch.int64, device=psi.device)

    psi = Z(psi, L, i, indice)
    psi = X(psi, L, i, indice)

    return 1j * psi


def Had(psi, L, i, indice = None):

    if indice is None:
        indice = torch.arange(2**L, dtype=torch.int64, device=psi.device)

    Xpi = X(psi, L, i, indice)
    Zpi = Z(psi, L, i, indice)

    return (Xpi + Zpi) / torch.sqrt(torch.tensor(2.0, device=device))


def S(psi, L, i, indice = None):

    if indice is None:
        indice = torch.arange(2**L, dtype=torch.int64, device=psi.device)


    if psi.dim() == 2:
        indice_z = indice.unsqueeze(1)
    else:
        indice_z = indice

    Spsi = (1 + ((indice_z >> i) & 1) * (1j - 1)) * psi

    return Spsi


def Rx(psi, theta, L, i, indice = None):

    ctheta = torch.cos(torch.tensor(theta / 2, device=psi.device))
    istheta = 1j * torch.sin(torch.tensor(theta / 2, device=psi.device))

    Xpsi = X(psi, L, i, indice)

    return ctheta * psi - istheta * Xpsi


def Ry(psi, theta, L, i, indice = None):

    ctheta = torch.cos(torch.tensor(theta / 2, device=psi.device))
    istheta = 1j * torch.sin(torch.tensor(theta / 2, device=psi.device))

    Ypsi = Y(psi, L, i, indice)

    return ctheta * psi - istheta * Ypsi


def Rz(psi, theta, L, i, indice = None):

    ctheta = torch.cos(torch.tensor(theta / 2, device=psi.device))
    istheta = 1j * torch.sin(torch.tensor(theta / 2, device=psi.device))

    Zpsi = Z(psi, L, i, indice)

    return ctheta * psi - istheta * Zpsi



def CNOT(psi, L, control, target, indice = None):

    if indice is None:
        indice = torch.arange(2**L, dtype=torch.int64, device=psi.device)

    novo_indice =  (((indice >> control) & 1) << target) ^ indice

    return psi[novo_indice]


def CZ(psi, L, control, target):

    if indice is None:
        indice = torch.arange(2**L, dtype=torch.int64, device=psi.device)

    indice_z = ((indice >> control) & (indice >> target)) & 1
    factor = 1 - 2 * indice_z
    
    return factor.unsqueeze(1) * psi if psi.dim() == 2 else factor * psi


def XX(psi, L, i, j, indice = None):

    if indice is None:
        indice = torch.arange(2**L, dtype=torch.int64, device=psi.device)

    novo_indice = indice ^ ((1 << i) | (1 << j))

    return psi[novo_indice]


def ZZ(psi, L, i, j, indice = None):

    if indice is None:
        indice = torch.arange(2**L, dtype=torch.int64, device=psi.device)

    if psi.dim() == 2:
        indice_z = indice.unsqueeze(1)
    else:
        indice_z = indice

    novo_indice = 1 - 2 * (((indice_z >> i) ^ (indice_z >> j)) & 1)

    return psi * novo_indice


def YY(psi, L, i, j, indice = None):

    if indice is None:
        indice = torch.arange(2**L, dtype=torch.int64, device=psi.device)

    psi = ZZ(psi, L, i, j, indice)
    psi = XX(psi, L, i, j, indice)

    return - psi