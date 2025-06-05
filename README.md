# qsim

Este repositório reúne funções implementadas em PyTorch para simulação de sistemas quânticos discretos, com foco em operadores de simetria, estados iniciais, Hamiltonianos e evolução temporal de vetores de estado.

## Estrutura dos Módulos

- `qsim/`
  - **`states.py`**: define funções para criação de estados iniciais (autoestados da base Z, base X, superposição uniforme, etc.).
  - **`hamiltonians.py`**: define os Hamiltonianos baseados em operadores de Pauli e interações de dois corpos (X, Y, Z, XX, YY).
  - **`evolution.py`**: implementa operadores de evolução temporal unitária $\exp(-i H \theta)$ com diferentes tipos de Hamiltonianos.
  - **`bitops.py`**: implementa operações de simetria (translação, inversão, reflexão) e funções auxiliares baseadas em manipulação de bits.
  - **`config.py`**: define o `device` e `dtype` utilizados globalmente nos módulos.

## Requisitos

- Python 3.8+
- PyTorch

Instale as dependências com:

```bash
pip install torch
```
