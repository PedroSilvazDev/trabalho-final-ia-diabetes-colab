Disciplina de Inteligência Artificial , Professor Munif , Unicesumar 2026

# Trabalho Final - Predição de Diabetes (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PedroSilvazDev/trabalho-final-ia-diabetes-colab/blob/main/trabalho_final_ia_colab.ipynb)

Versão do trabalho otimizada para **Google Colab**. Mesmo dataset, mesmos modelos (KNN + SVM) e mesma análise do repositório principal.

Repositório PC/local: [trabalho-final-ia-diabetes](https://github.com/PedroSilvazDev/trabalho-final-ia-diabetes)

## Integrantes

- Pedro Henrique da Silva - RA: 23021607-2
- Victor Hugo Rodrigues de Oliveira - RA: 23418156-2
- Victor Hungo Silva Garcia - RA: 23030968-2

## Como executar no Colab

### Opção 1 — Botão (recomendado)

1. Clique no badge **Open In Colab** acima
2. No menu: **Runtime → Run all** (ou `Ctrl+F9`)
3. Aguarde o treinamento e os gráficos aparecerem no notebook

### Opção 2 — Manual

1. Acesse [Google Colab](https://colab.research.google.com/)
2. **File → Open notebook → GitHub**
3. Cole: `PedroSilvazDev/trabalho-final-ia-diabetes-colab`
4. Abra `trabalho_final_ia_colab.ipynb`
5. **Runtime → Run all**

## O que o notebook faz

- Baixa o dataset Pima Indians Diabetes
- Preprocessa os dados (imputação + padronização)
- Treina **KNN** (Parte 1) e **SVM** (Parte 2)
- Exibe métricas, matrizes de confusão e gráficos comparativos
- Apresenta conclusão com os resultados

## Estrutura

```text
trabalho-final-ia-diabetes-colab/
├── trabalho_final_ia_colab.ipynb   # notebook principal
├── src/                            # módulos reutilizados
├── data/                           # dataset (baixado automaticamente)
└── README.md
```

## Apresentação

Para mostrar o treinamento ao vivo na apresentação, abra o Colab, compartilhe a tela e execute **Run all**.
