# qsim

Este repositório contém funções implementadas em PyTorch para simulação de sistemas quânticos baseados em vetores de estado. O objetivo é manipular e evoluir os estados diretamente sobre os tensores que os representam, sem recorrer a operações matriciais completas. Isso permite simulações mais eficientes em termos de memória e desempenho, especialmente para sistemas com muitos qubits.


## Estrutura dos Módulos

- `qsim/`
  - **`states.py`**: define funções para criação de estados iniciais (autoestados da base Z, base X, superposição uniforme, etc.).
  - **`operators.py`**: define operadores quânticos básicos (X, Y, Z, Hadamard, S, Rx, Ry, Rz, CNOT, CZ, XX, YY, ZZ).
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
