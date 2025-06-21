import torch
from .. import diagonal as qd
from .. import bitops as qb
from .. import config as qconfig

def indice_tsp(num_cidades : int, cidade_i : int, posicao : int, fixar_cidade : bool = False):
    """
    Retorna o índice correspondente à variável associada à cidade `cidade_i` 
    na posição `posicao` do percurso, para o problema do caixeiro viajante (TSP).

    Se `fixar_cidade` for True, assume-se que a cidade 0 está fixada na posição 0 
    e não é representada por variáveis.
    
    Parâmetros:
        num_cidades (int): número total de cidades
        cidade_i (int): índice da cidade (0 a num_cidades-1)
        posicao (int): posição no ciclo (0 a num_cidades-1)
        fixar_cidade (bool): se a cidade 0 está fixada na posição 0 (padrão: False)
    
    Retorna:
        int: índice da variável correspondente, ou -1 se for cidade/posição fixada
    """

    # Garante que os índices estão dentro dos limites
    cidade_i %= num_cidades
    posicao %= num_cidades

    if fixar_cidade:
        # Se a cidade 0 ou posição 0 estiver presente, não há variável associada
        if cidade_i == 0 or posicao == 0:
            return -1
        
        # Mapeia o par (cidade_i, posicao) para um índice único
        # considerando que a cidade 0 e a posição 0 não têm variáveis associadas
        return ((num_cidades - 1) ** 2) - ((num_cidades - 1) * (cidade_i - 1) + (posicao - 1)) - 1
    else:
        # Mapeia o par (cidade_i, posicao) para um índice único
        # quando todas as cidades e posições têm variáveis associadas
        return (num_cidades ** 2) - (num_cidades * cidade_i + posicao) - 1



def H_tsp(num_cidades : int, w, A : float = 10, B : float = 1, fixar_cidade : bool =False):
    """
    Constrói o Hamiltoniano do problema do caixeiro viajante (TSP) no formato binário,
    penalizando violações das restrições e atribuindo custo às transições entre cidades.

    Parâmetros:
        num_cidades (int): número total de cidades
        w (torch.Tensor): matriz de pesos (distâncias) entre as cidades
        A (float): penalidade para violações das restrições
        B (float): peso do custo das arestas no ciclo
        fixar_cidade (bool): se True, fixa a cidade 0 na posição 0

    Retorna:
        torch.Tensor: vetor de dimensão 2^L contendo os valores da diagonal principal 
        do Hamiltoniano, onde cada entrada representa o valor da função de custo 
        para uma configuração binária do circuito.
    """

    # Número total de variáveis booleanas (qubits)
    if fixar_cidade:
        L = (num_cidades - 1) ** 2
    else:
        L = num_cidades ** 2

    # Inicializa o Hamiltoniano com zeros
    H = torch.zeros(2 ** L, dtype=qconfig.dtype, device=qconfig.device)

    # Define o índice inicial (pula a cidade e posição 0 se fixar_cidade for True)
    inicio = 1 if fixar_cidade else 0

    # Restrição: cada cidade deve ser visitada exatamente uma vez
    for i in range(inicio, num_cidades):
        termo = torch.ones(2 ** L, dtype=qconfig.dtype, device=qconfig.device)
        for p in range(inicio, num_cidades):
            k = indice_tsp(num_cidades, i, p, fixar_cidade)
            termo -= qd.cadeia_numero(L, [k])  # variável booleana ligada à cidade i na posição p
        H += A * (termo ** 2)  # penalidade quadrática

    # Restrição: cada posição no ciclo deve conter exatamente uma cidade
    for p in range(inicio, num_cidades):
        termo = torch.ones(2 ** L, dtype=qconfig.dtype, device=qconfig.device)
        for i in range(inicio, num_cidades):
            k = indice_tsp(num_cidades, i, p, fixar_cidade)
            termo -= qd.cadeia_numero(L, [k])  # variável booleana ligada à cidade i na posição p
        H += A * (termo ** 2)  # penalidade quadrática

    # Custo: soma dos pesos das arestas entre cidades consecutivas no ciclo
    for i in range(num_cidades):
        for j in range(num_cidades):
            if i != j:
                for p in range(num_cidades):
                    # índice da cidade i na posição p
                    ki = indice_tsp(num_cidades, i, p, fixar_cidade)
                    # índice da cidade j na posição p+1 (ciclo fechado)
                    kj = indice_tsp(num_cidades, j, (p + 1) % num_cidades, fixar_cidade)

                    if fixar_cidade:
                        # Tratamento especial para as posições fixadas
                        if i == 0 and p == 0:
                            H += qd.cadeia_numero(L, [kj], w[i, j] * B)
                        elif j == 0 and (p + 1) % num_cidades == 0:
                            H += qd.cadeia_numero(L, [ki], w[i, j] * B)
                        elif i != 0 and j != 0 and p != 0 and (p + 1) % num_cidades != 0:
                            H += qd.cadeia_numero(L, [ki, kj], w[i, j] * B)
                    else:
                        H += qd.cadeia_numero(L, [ki, kj], w[i, j] * B)

    return H



def ciclo_hamiltoniano(estado : int, num_cidades : int, fixar_cidade = False):
    """
    Dado um inteiro `estado` representando uma configuração binária de um possível
    percurso no TSP (formulado como QUBO), reconstrói o ciclo correspondente.

    Parâmetros:
        estado (int): configuração binária representando um estado base (|x⟩)
        num_cidades (int): número de cidades
        fixar_cidade (bool): se True, assume que a cidade 0 está fixada na posição 0

    Retorna:
        list[int]: ciclo de cidades visitadas na ordem definida pela configuração
    """

    if fixar_cidade:
        L = (num_cidades-1) ** 2
        num_cidades -= 1
    else:
        L = num_cidades ** 2

    ciclo = [-1] * n

    filtro = (1 << num_cidades) - 1

    for cidade in range(n):
        posicao = estado & filtro
      
        if qb.possui_um_bit_1(posicao):
            posicao = posicao.bit_length() - 1
        else:
            posicao = -1
          
        if posicao != -1:
            ciclo[posicao] = (cidade + 1) if fixar_cidade else cidade
          
        estado >>= num_cidades

    if fixar_cidade:
        ciclo = [0] + ciclo

    return ciclo
