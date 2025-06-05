import torch
from .operators import Z, X, Y, ZZ, XX, YY



def Hx(psi, L, indice):
    Hxpsi = torch.zeros_like(psi)

    for i in range(L):
        Hxpsi += X(psi, L, i, indice)
    return Hxpsi


def Hy(psi, L, indice):
    Hypsi = torch.zeros_like(psi)

    for i in range(L):
        Hypsi += Y(psi, L, i, indice)
    return Hypsi


def Hz(psi, L, indice):
    Hzpsi = torch.zeros_like(psi)

    for i in range(L):
        Hzpsi += Z(psi, L, i, indice)
    return Hzpsi


def Hxx(psi, L, w, indice):
    Hxxpsi = torch.zeros_like(psi)

    for i in range(L):
        for j in range(i + 1, L):
            if w[i, j] != 0:
                Hxxpsi += XX(psi, L, i, j, indice)

    return Hxxpsi


def Hyy(psi, L, w, indice):
    Hyypsi = torch.zeros_like(psi)

    for i in range(L):
        for j in range(i + 1, L):
            if w[i, j] != 0:
                Hyypsi += YY(psi, L, i, j, indice)
    
    return Hyypsi