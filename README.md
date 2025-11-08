# 🤖 Análise de Sentimentos em Pull Requests

Este projeto contém um script Python que utiliza a biblioteca `transformers` da Hugging Face para analisar o sentimento dos comentários nos 100 Pull Requests fechados mais recentes de um repositório GitHub.

O script:

1. Busca os 100 PRs fechados mais recentes do repositório configurado.
2. Para cada PR, coleta todos os seus comentários.
3. Usa múltiplos modelos de linguagem (listados no próprio script) para classificar o sentimento de cada comentário (positivo, negativo, neutro).
4. Gera um arquivo `.csv` separado para cada modelo com os resultados da análise.

---

## 🚀 Como Rodar o Projeto

Siga estas etapas para configurar e executar o script.

**Nota Importante:** A estrutura de pastas esperada (baseada na imagem fornecida) é que este script esteja em uma pasta `src/`. Todos os comandos abaixo devem ser executados **de dentro da pasta `src`**.

```

Evolucao_Software_2025-2_screenpipe/
├── src/  \<-- Você deve estar aqui
│   ├── .env.example
│   ├── .gitignore
│   ├── analisar\_sentimentos\_prs.py
│   ├── requirements.txt
│   └── ...
└── ...

```

### 1. Pré-requisitos

-   Python 3.8+
-   Git (para clonar o repositório)

### 2. Configuração do Ambiente

**1. Clone o repositório e entre na pasta `src`:**

```bash
git clone https://github.com/argologabriel/Evolucao_Software_2025-2_screenpipe
cd Evolucao_Software_2025-2_screenpipe/src
```

**2. Crie e Ative um Ambiente Virtual:**
É altamente recomendado usar um ambiente virtual (`venv`) para isolar as dependências.

```bash
# Crie o ambiente (só precisa fazer isso uma vez)
python -m venv venv

# Ative o ambiente (precisa fazer toda vez que for rodar)
# No Windows:
.\venv\Scripts\activate
# No macOS / Linux:
source venv/bin/activate
```

**3. Instale as Dependências:**
Com o ambiente ativado (você verá `(venv)` no seu terminal), instale as bibliotecas necessárias.

```bash
pip install -r requirements.txt
```

_(Isso pode demorar um pouco, pois baixará as bibliotecas `transformers` e `torch`/`tensorflow`)._

### 3\. Variáveis de Ambiente (Configuração)

O script precisa de "segredos" (seu token do GitHub) e configurações (o nome do repositório) para rodar.

**1. Crie seu Token do GitHub:**
Você precisa de um "Personal Access Token (classic)" para permitir que o script acesse a API do GitHub sem ser bloqueado por limites de taxa.

-   Vá para sua conta do GitHub: **Settings** \> **Developer settings** \> **Personal access tokens** \> **Tokens (classic)**.
-   Clique em **Generate new token** (e depois "Generate new token (classic)").
-   Dê um nome para o token (ex: `analise-prs-script`).
-   Em **Select scopes**, marque a permissão **`public_repo`** (localizada dentro da seção `repo`). Isso é suficiente para ler repositórios públicos.
-   Clique em **Generate token**.
-   **Copie o token imediatamente\!** (ex: `ghp_...`). Você não o verá novamente.

**2. Configure o arquivo `.env`:**
Dentro da pasta `src/`, faça uma cópia do arquivo de exemplo:

```bash
# No Windows
copy .env.example .env
# No macOS / Linux
cp .env.example .env
```

Abra o novo arquivo `.env` com um editor de texto e cole seu token. Ele deve ficar assim:

```ini
# Cole o token que você acabou de gerar
GITHUB_TOKEN = ghp_SEUTOKENCOPIADOAQUI

# O script já está configurado para usar este repositório:
REPO_OWNER = mediar-ai
REPO_NAME = screenpipe
```

_Salve e feche o arquivo._

### 4\. Executando o Script

Com o ambiente ativado (`(venv)`) e o arquivo `.env` configurado, você está pronto para rodar:

```bash
python analisar_sentimentos_prs.py
```

O script exibirá o progresso no terminal:

-   Primeiro, ele coletará os PRs.
-   Depois, ele fará um loop por cada modelo da lista. Na primeira execução, ele **baixará cada modelo** (pode demorar).
-   Você verá barras de progresso (`tqdm`) para a análise de cada modelo.
-   Ao final de cada modelo, ele imprimirá um resumo dos sentimentos.

---

## 📊 Saída (Resultados)

O script criará automaticamente uma pasta `docs/` na raiz do seu projeto (`EVOLUCAO/`) e salvará os CSVs lá.

A estrutura final ficará assim:

```
EVOLUCAO/
├── docs/   <-- ✅ SEUS RESULTADOS ESTÃO AQUI
│   ├── sentimentos_roberta_cardiff.csv
│   ├── sentimentos_distilbert_en.csv
│   └── sentimentos_distilbert_multi.csv
│
└── src/
    ├── .env
    ├── .env.example
    ├── .gitignore
    ├── analisar_sentimentos_prs.py
    ├── requirements.txt
    └── venv/
```

Cada arquivo `.csv` conterá as seguintes colunas:

-   `PR_Number`: O número do Pull Request.
-   `PR_Title`: O título do Pull Request.
-   `Comment`: O texto do comentário analisado.
-   `Sentiment`: A etiqueta de sentimento (ex: `NEGATIVE`, `NEUTRAL`, `POSITIVE`).
-   `Score`: A confiança do modelo (de 0.0 a 1.0) na sua classificação.

---

## 🔧 Customização

-   **Para analisar um repositório diferente:** Altere os valores de `REPO_OWNER` e `REPO_NAME` no seu arquivo `.env`.
-   **Para testar outros modelos:** Edite o dicionário `MODEL_LIST` diretamente no arquivo `analisar_sentimentos_prs.py`. Adicione ou remova pares de `"apelido": "nome-do-modelo-no-huggingface"`.
