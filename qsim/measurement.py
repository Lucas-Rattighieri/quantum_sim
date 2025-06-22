import torch

def medir_state_vector(psi: torch.Tensor, nshots: int, retornar_porcentagem: bool = False):
    """
    psi (torch.Tensor): vetor de estado, representado como tensor complexo.
    nshots (int): número de amostras (medidas)
    retornar_porcentagem (bool): se True, retorna a frequência relativa (porcentagem) de cada estado medido

    Retorna:
        estados (torch.Tensor): índices dos estados da base computacional que foram medidos.
        valores (torch.Tensor): contagens absolutas (se retornar_porcentagem=False)
                               ou porcentagens (se retornar_porcentagem=True),
                               na mesma ordem de 'estados'.
    """

    probs = psi.abs()**2
    medidas = torch.multinomial(probs, num_samples=nshots, replacement=True)
    estados, contagens = torch.unique(medidas, return_counts=True)

    if retornar_porcentagem:
        porcentagens = contagens.float() / nshots
        return estados, porcentagens

    return estados, contagens


def prob_estado_base(psi : torch.Tensor, estado : int):
    """
    Calcula a probabilidade de medir o vetor base computacional 'estado' no estado quântico 'psi'.

    Parâmetros:
        psi (torch.Tensor): vetor de estado quântico (complexo).
        estado (int): índice do estado base computacional.

    Retorna:
        (float): probabilidade de obter o 'estado' na medição, ou seja, |psi[estado]|^2.
    """
    return float(torch.abs(psi[estado]) ** 2)
