import os
import requests
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# ===============================
# CONFIGURAÇÕES INICIAIS
# ===============================

# Repositório alvo
OWNER = "mediar-ai"
REPO = "screenpipe"

# Token do GitHub (opcional)
TOKEN = os.getenv("GITHUB_TOKEN")

# Cabeçalhos da requisição
headers = {"Authorization": f"token {TOKEN}"} if TOKEN else {}

# Modelo de sentimento
sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# ===============================
# FUNÇÃO PARA PEGAR PRs
# ===============================

def get_closed_pull_requests(owner, repo, limit=100):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    params = {"state": "closed", "per_page": 100, "page": 1}
    prs = []

    while len(prs) < limit:
        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        prs.extend(data)
        params["page"] += 1

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

print("🔍 Coletando Pull Requests fechados...")
prs = get_closed_pull_requests(OWNER, REPO, limit=100)

results = []

print("🧠 Analisando sentimentos dos comentários...\n")

for pr in tqdm(prs):
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
            sentiment = sentiment_model(comment[:512])[0]  # limite de tokens
            results.append({
                "PR_Number": pr_number,
                "PR_Title": pr_title,
                "Comment": comment,
                "Sentiment": sentiment["label"],
                "Score": sentiment["score"]
            })
        except Exception as e:
            print(f"Erro ao analisar PR #{pr_number}: {e}")

# ===============================
# CRIAR DATAFRAME E RELATÓRIO
# ===============================

df = pd.DataFrame(results)
df.to_csv("sentimentos_screenpipe.csv", index=False, encoding="utf-8-sig")

# Resumo geral
summary = df["Sentiment"].value_counts(normalize=True) * 100
print("\n📊 Resumo de sentimentos (%):")
print(summary)

print("\n✅ Arquivo salvo como 'sentimentos_screenpipe.csv'")
