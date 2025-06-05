import torch
from .operators import Z, X, Y, ZZ, XX, YY


def expHx(psi, theta, L, indice):
    ctheta = torch.cos(torch.tensor(theta, device=psi.device))
    istheta = 1j * torch.sin(torch.tensor(theta, device=psi.device))

    for a in range(L):
        cpsi = psi * ctheta
        isXpsi = X(psi, L, a, indice) * istheta
        psi = cpsi - isXpsi

    return psi


def expHy(psi, theta, L, indice):
    ctheta = torch.cos(torch.tensor(theta, device=psi.device))
    istheta = 1j * torch.sin(torch.tensor(theta, device=psi.device))

    for a in range(L):
        cpsi = psi * ctheta
        isXpsi = Y(psi, L, a, indice) * istheta
        psi = cpsi - isXpsi

    return psi


def expHz(psi, theta, L, indice):
    ctheta = torch.cos(torch.tensor(theta, device=psi.device))
    istheta = 1j * torch.sin(torch.tensor(theta, device=psi.device))

    for a in range(L):
        cpsi = psi * ctheta
        isXpsi = Z(psi, L, a, indice) * istheta
        psi = cpsi - isXpsi

    return psi


def expHxx(psi, theta, L, w, indice):
    ctheta = torch.cos(torch.tensor(theta, device=psi.device))
    istheta = 1j * torch.sin(torch.tensor(theta, device=psi.device))

    for i in range(L):
        for j in range(i + 1, L):
            if w[i, j] != 0:
                cpsi = psi * ctheta
                isXXpsi = XX(psi, L, i, j, indice) * istheta
                psi = cpsi - isXXpsi

    return psi


def expHyy(psi, theta, L, w, indice):
    ctheta = torch.cos(torch.tensor(theta, device=psi.device))
    istheta = 1j * torch.sin(torch.tensor(theta, device=psi.device))

    for i in range(L):
        for j in range(i + 1, L):
            if w[i, j] != 0:
                cpsi = psi * ctheta
                isYYpsi = YY(psi, L, i, j, indice) * istheta
                psi = cpsi - isYYpsi

    return psi


def expHxy(psi, theta, L, w, indice):
    ctheta = torch.cos(torch.tensor(theta, device=psi.device))
    istheta = 1j * torch.sin(torch.tensor(theta, device=psi.device))

    for i in range(L):
        for j in range(i + 1, L):
            if w[i, j] != 0:
                cpsi = psi * ctheta
                
                isXXpsi = XX(psi, L, i, j, indice) * istheta
                psi = cpsi - isXXpsi

                isYYpsi = YY(psi, L, i, j, indice) * istheta
                psi = cpsi - isYYpsi

    return psi