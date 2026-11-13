# Rede Neural Artificial para classificação do Iris

## Sobre o projeto

Este projeto acadêmico foi desenvolvido para a disciplina de **Inteligência Artificial**, do curso de **Ciência da Computação** do Centro Universitário Presidente Antônio Carlos (UNIPAC Barbacena).

O objetivo é implementar, treinar e avaliar uma Rede Neural Artificial capaz de classificar flores do conjunto de dados **Iris**. A rede é construída do zero: as operações de propagação direta, cálculo da perda, retropropagação e atualização dos pesos são implementadas manualmente com **NumPy**.

O Scikit-Learn é utilizado somente para carregar o conjunto Iris e realizar a divisão entre dados de treinamento e teste.


## Conjunto de dados

O conjunto Iris possui 150 amostras de flores, distribuídas em três espécies:

- `setosa`
- `versicolor`
- `virginica`

Cada amostra possui quatro características numéricas:

1. comprimento da sépala;
2. largura da sépala;
3. comprimento da pétala;
4. largura da pétala.

Os dados são divididos em 80% para treinamento e 20% para teste. As características são padronizadas usando a média e o desvio padrão do conjunto de treinamento. Os rótulos são convertidos para vetores One-Hot com três posições.

## Arquitetura da rede

A rede possui a arquitetura **4 → 8 → 3**:

- **Camada de entrada:** recebe as quatro características do Iris;
- **Camada escondida:** possui oito neurônios e utiliza a função de ativação ReLU;
- **Camada de saída:** possui três neurônios, um para cada espécie, e utiliza Softmax para produzir probabilidades.

Os pesos são inicializados aleatoriamente com inicialização de He, e os vieses começam com zero. Uma semente fixa (`seed=42`) é usada para tornar o resultado reproduzível.

## Como o treinamento funciona

Em cada época, o programa realiza as seguintes etapas:

1. **Forward Pass:** calcula as ativações da camada escondida e da camada de saída;
2. **Loss:** calcula a perda usando Entropia Cruzada (`Cross-Entropy`);
3. **Backward Pass:** calcula os gradientes por retropropagação, utilizando a regra da cadeia;
4. **Atualização:** atualiza pesos e vieses com Gradiente Descendente.

O treinamento utiliza 1000 épocas e taxa de aprendizado igual a `0.05`. Ao final, o modelo classifica o conjunto de teste e calcula a acurácia.

## Resultado esperado

Com a divisão dos dados e a inicialização definidas no código, a execução produz uma acurácia próxima de:

```text
Acurácia final no conjunto de teste: 96.67%
```

O valor pode variar caso a semente, a divisão dos dados, a arquitetura ou os hiperparâmetros sejam alterados.

## Como executar

### 1. Instalar as dependências

Recomenda-se utilizar um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scikit-learn
```

No Windows, a ativação pode ser feita com:

```powershell
.venv\Scripts\activate
```

### 2. Executar o programa

```bash
python main.py
```

O programa exibirá a evolução da perda durante o treinamento e a acurácia final no conjunto de teste.

## Arquivos do projeto

| Arquivo | Descrição |
|---|---|
| [`main.py`](main.py) | Código principal do projeto. Carrega e prepara os dados, define as funções de ativação, implementa a classe da rede neural, realiza o treinamento e calcula a acurácia. |
| [`README.md`](README.md) | Documentação do projeto, com seu objetivo acadêmico, funcionamento, instruções de execução e descrição dos arquivos. |
| `.venv/` | Ambiente virtual local criado para instalar as dependências. É um diretório de configuração do ambiente e não faz parte da implementação da rede neural. |

