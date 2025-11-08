import os
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import pipeline

# Carregar variáveis de ambiente
load_dotenv()

# Repositório alvo
OWNER = os.getenv("REPO_OWNER")
REPO = os.getenv("REPO_NAME")

# Token do GitHub
TOKEN = os.getenv("GITHUB_TOKEN")

# Cabeçalhos da requisição
headers = {"Authorization": f"token {TOKEN}"} if TOKEN else {}

# Lista de modelos de sentimento
MODEL_LIST = {
    "roberta_cardiff": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "distilbert_en": "distilbert-base-uncased-finetuned-sst-2-english",
    "distilbert_multi": "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
}

# ===============================
# FUNÇÃO PARA PEGAR PRs
# ===============================

def get_closed_pull_requests(owner, repo, limit=100):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    # Começa na página 1 para pegar os mais recentes
    params = {"state": "closed", "per_page": 100, "page": 1}
    prs = []

    while len(prs) < limit:
        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        prs.extend(data)
        # Incrementa a página para a próxima iteração
        params["page"] += 1 

    # Retorna apenas o limite solicitado
    return prs[:limit]

# ===============================
# FUNÇÃO PARA PEGAR COMENTÁRIOS DE UM PR
# ===============================

def get_pr_comments(owner, repo, pr_number):
    comments_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    r = requests.get(comments_url, headers=headers)
    if r.status_code == 200:
        return [c["body"] for c in r.json()]
    return []

# ===============================
# EXECUÇÃO PRINCIPAL
# ===============================

# Define o diretório de saída
OUTPUT_DIR = "../docs"

# 1. Criar o diretório de saída se não existir
print(f"📂 Verificando/criando diretório de saída: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Coletar PRs (apenas uma vez)
print("🔍 Coletando Pull Requests fechados...")
prs = get_closed_pull_requests(OWNER, REPO, limit=100)
print(f"✅ {len(prs)} PRs coletados.")

# 3. Loop principal sobre o DICIONÁRIO de modelos
for short_name, model_path in MODEL_LIST.items():
    print(f"\n{'='*60}")
    # Usa o apelido e o nome completo no log
    print(f"🤖 Processando modelo: {short_name} ({model_path})")
    print(f"{'='*60}")

    try:
        # Carrega o modelo usando o nome completo (model_path)
        print("Loading model...")
        sentiment_model = pipeline("sentiment-analysis", model=model_path)
    except Exception as e:
        print(f"❌ Erro ao carregar o modelo {model_path}: {e}. Pulando para o próximo.")
        continue

    results = []
    print("🧠 Analisando sentimentos dos comentários...\n")

    # Usa o apelido (short_name) para a barra de progresso
    for pr in tqdm(prs, desc=f"Analisando com {short_name}"):
        pr_number = pr["number"]
        pr_title = pr["title"]
        comments = get_pr_comments(OWNER, REPO, pr_number)

        if not comments:
            results.append({
                "PR_Number": pr_number,
                "PR_Title": pr_title,
                "Comment": "",
                "Sentiment": "neutral",
                "Score": 0.0
            })
            continue

        for comment in comments:
            try:
                sentiment = sentiment_model(comment[:512])[0]
                results.append({
                    "PR_Number": pr_number,
                    "PR_Title": pr_title,
                    "Comment": comment,
                    "Sentiment": sentiment["label"],
                    "Score": sentiment["score"]
                })
            except Exception as e:
                # Adiciona o apelido ao log de erro
                print(f"Erro ao analisar PR #{pr_number} com modelo {short_name}: {e}")

    # ===============================
    # CRIAR DATAFRAME E RELATÓRIO
    # ===============================
    
    if not results:
        print(f"⚠️ Nenhum resultado gerado para o modelo {short_name}.")
        continue

    df = pd.DataFrame(results)

    # Nome do arquivo de saída usando o apelido do modelo
    output_filename = f"sentimentos_{short_name}.csv"
    
    # Criar o caminho completo do arquivo
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Salvar o CSV no diretório de saída
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Resumo geral para este modelo
    summary = df["Sentiment"].value_counts(normalize=True) * 100
    print(f"\n📊 Resumo de sentimentos (%) para {short_name}:")
    print(summary)

    print(f"\n✅ Arquivo salvo como '{output_path}'")