# Desafio Tecnico MLE

Este repositório contém o desenvolvimento de um projeto de uma API de previsão de atraso de chegada de voo, focado em dados, incluindo treinamento de modelos e documentação. Abaixo estão as pastas principais e suas respectivas funções.

## Estrutura do Repositório

### 1. **data/**
   Esta pasta contém o dataset em formato .csv

### 2. **notebook/**
   Esta pasta contém notebooks Jupyter usados para:
   - **Exploração de Dados**: Análises exploratórias e visualizações dos dados.
   - **Teste de Modelos**: Experimentos com diferentes algoritmos de machine learning.

### 3. **src/**
   Aqui estão os arquivos de código-fonte do projeto. Esta pasta inclui:
   - **Modelos**: Código para definir e treinar modelos de machine learning.
   - **API**: Scripts que implementam a API
   - **Documentação**: Notas e comentários no código para facilitar a compreensão.

### 4. **tests/**
   Esta pasta contém os testes automatizados do projeto. Aqui você encontrará:
   - **Testes de unidade**: Scripts que garantem que cada componente do código funcione corretamente.
   - **Testes de integração**: Verificações para garantir que diferentes partes do sistema interajam conforme esperado.

## Como Executar o Projeto

1. Clone este repositório:
   ```bash
   git clone https://github.com/lhupalo/desafio-mle.git
2. Entre no diretório clonado e aça o build do docker compose
   ```bash
   docker compose build
3. Suba o container
   ```bash
   docker compose up -d
4. A API estará disponivel no localhost:8080
