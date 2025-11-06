**Análise de Sentimentos em Pull Requests do GitHub**
=====================================================

Este projeto foi desenvolvido para a disciplina de Evolução de Software 2025-2.

O objetivo é analisar a evolução de um projeto de software open-source através da análise de sentimentos dos comentários deixados nos Pull Requests (PRs). O script automatiza a coleta de dados do GitHub e a análise usando modelos de linguagem do Hugging Face.

**Tecnologias Utilizadas**
--------------------------

*   **Python**
    
*   **Requests:** Para fazer chamadas à API REST do GitHub e coletar os dados.
    
*   **Pandas:** Para organizar os dados dos comentários numa tabela e salvar em arquivos CSV.
    
*   **Hugging Face transformers:** Para baixar e executar os modelos de análise de sentimentos.
    
*   **PyTorch:** A biblioteca que serve de "motor" para os modelos do Hugging Face.
    
*   **Google Colab:** Utilizado como ambiente de execução para contornar limitações de Memória RAM e acelerar o processamento com GPU gratuita.
    

**Como Funciona:**
------------------

O script coleta\_e\_analise.py executa o processo completo em duas etapas principais:

### **Parte 1: Coleta de Dados do GitHub**

1.  O script autentica-se na API do GitHub usando um Token de Acesso Pessoal.
    
2.  Ele busca os 100 Pull Requests (PRs) fechados mais recentes do repositório (mediar-ai/screenpipe).
    
3.  Para cada um desses 100 PRs, ele faz novas chamadas à API para buscar _todos_ os comentários associados (tanto os comentários gerais do PR quanto os comentários de revisão de código).
    
4.  É aplicada uma verificação para lidar com "usuários fantasmas" (usuários deletados cujo campo user é None).
    
5.  Todos os comentários válidos são limpos (removendo os vazios) e salvos num arquivo de backup: github\_comments\_mediar-ai\_screenpipe.csv.
    

### **Parte 2: Análise de Sentimentos**

1.  O script carrega o arquivo CSV gerado na Parte 1.
    
2.  Para evitar erros de memória (como o bus error), ele **não carrega os 3 modelos de uma vez**. Em vez disso, ele entra num loop:
    
3.  **Carrega** o Modelo 1 (ex: distilbert).
    
4.  **Aplica** o Modelo 1 em _cada linha_ (comentário) do CSV.
    
5.  **Liberta** o Modelo 1 da memória (gc.collect()).
    
6.  **Carrega** o Modelo 2 (ex: twitter-roberta).
    
7.  **Aplica** o Modelo 2 em cada comentário.
    
8.  **Liberta** o Modelo 2 da memória.
    
9.  Repete o processo para o Modelo 3.
    
10.  No final, salva um novo arquivo, sentimentos\_analisados.csv, contendo os comentários originais e as novas colunas com os resultados de cada modelo.
    

**Como os Modelos Analisam os Comentários?**
--------------------------------------------

Esta é a parte mais importante. Os modelos de linguagem (LLMs) **não sabem o que é um arquivo CSV**. Eles só processam texto.

A "magia" acontece através do nosso script Python, que atua como um gestor:

1.  **O Script Lê o CSV:** A biblioteca pandas abre o sentimentos\_analisados.csv e o coloca numa tabela na memória.
    
2.  **O Script Pega um Comentário:** O script vai à Tabela, linha 1, e pega o texto da coluna comment\_body.
    
3.  **O Script Envia o Texto para o Modelo:** O script entrega _apenas esse texto_ ao modelo de análise de sentimentos (ex: distilbert) que está carregado na memória.
    
4.  **O Modelo Calcula:** O modelo é uma rede neural que foi treinada em milhões de textos. Ele transforma as palavras em números (tokens) e, com base no seu treino, calcula a probabilidade de o texto ser positivo, negativo ou neutro.
    
5.  **O Modelo Retorna uma Resposta:** O modelo devolve uma resposta simples, como: {'label': 'POSITIVE', 'score': 0.99}.
    
6.  **O Script Guarda a Resposta:** O nosso script (pandas) pega nesta resposta e escreve POSITIVE na nova coluna (distilbert\_label) para a linha 1.
    
7.  **Repete:** O script passa para a linha 2 da tabela, pega o próximo comentário e repete todo o processo 517 vezes, uma para cada comentário e para cada modelo.
    

**Você pode p**ensar no **modelo** como um especialista que só sabe ler um pequeno trecho de texto e determinar como Positivo, Neutro ou Negativo. O **script Python** é o gestor que vai ao arquivo (o CSV), pega algum comentário em alguma célula, manda para o especialista (modelo), e recebe uma resposta em relação ao texto para escrever novamente no arquivo.

**Modelos Utilizados**
----------------------

Para este projeto, foram selecionados três modelos de análise de sentimentos do Hugging Face:

1.  **distilbert-base-uncased-finetuned-sst-2-english**
    
    *   **Descrição:** Um modelo "destilado" (versão mais leve) do BERT. É muito popular, rápido e oferece um ótimo equilíbrio entre performance e precisão para análise de sentimentos geral.
        
2.  **cardiffnlp/twitter-roberta-base-sentiment-latest**
    
    *   **Descrição:** Um modelo RoBERTa (uma versão melhorada do BERT) que foi treinado especificamente em milhões de tweets.
        
3.  **lxyuan/distilbert-base-multilingual-cased-sentiments-student**
    
    *   **Descrição:** Um modelo multilíngue. Foi escolhido para garantir que, caso o projeto screenpipe tivesse comentários em outros idiomas além do inglês, o modelo ainda conseguisse classificá-los corretamente.
        

**Como Executar**
-----------------

*   **Clonar o Repositório:**

    *  Bash git clone https://github.com/argologabriel/Evolucao\_Software\_2025-2\_screenpipe.git

    *  cd Evolucao\_Software\_2025-2

*   **Instalar as Dependências:**

    *  No Terminal: pip install pandas requests torch transformers
    
*   **Configurar o Token:**
    
    *   Abra o arquivo coleta\_e\_analise.py.
        
    *   Coloque o seu Token do Github na variável correspondente.
        
*   **Executar (Recomendado: Google Colab):**
    
    *   Devido ao alto uso de memória RAM, é recomendado fazer o upload deste script para o Google Colab e executá-lo num ambiente com "T4 GPU".
        
    *   A execução local pode ser muito lenta ou falhar por falta de memória (bus error).
        
*   **Ver os Resultados:**
    
    *   O script irá gerar o arquivo sentimentos\_analisados.csv com os resultados.