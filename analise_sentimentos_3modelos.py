import os
import requests
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# ===============================
# CONFIGURAÇÕES
# ===============================

OWNER = "mediar-ai"
REPO = "screenpipe"
TOKEN = os.getenv("GITHUB_TOKEN")
headers = {"Authorization": f"token {TOKEN}"} if TOKEN else {}

# 3 MODELOS
MODELOS = {
    "roberta_cardiff": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "roberta_jhartmann": "j-hartmann/sentiment-roberta-large-english-3-classes",
    "bert_sst2": "textattack/bert-base-uncased-SST-2"
}

# ===============================
# FUNÇÕES GITHUB
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


def get_pr_comments(owner, repo, pr_number):
    comments_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    r = requests.get(comments_url, headers=headers)
    if r.status_code == 200:
        return [c["body"] for c in r.json()]
    return []


# ===============================
# EXECUÇÃO PRINCIPAL POR MODELO
# ===============================

print("🔍 Coletando Pull Requests fechados...")
prs = get_closed_pull_requests(OWNER, REPO, limit=100)
print(f"✅ {len(prs)} PRs coletados.\n")

for nome, modelo in MODELOS.items():

    print(f"\n🧠 Iniciando análise com o modelo: {modelo}")
    sentiment_model = pipeline("sentiment-analysis", model=modelo)

    results = []

    for pr in tqdm(prs):
        pr_number = pr["number"]
        pr_title = pr["title"]
        comments = get_pr_comments(OWNER, REPO, pr_number)

        if not comments:
            results.append({
                "PR_Number": pr_number,
                "PR_Title": pr_title,
                "Comment": "",
                "Sentiment": "NEU",
                "Score": 0.0,
                "Model": nome
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
                    "Score": sentiment["score"],
                    "Model": nome
                })
            except Exception as e:
                print(f"⚠ Erro no PR #{pr_number}: {e}")

    # Salvar CSV por modelo
    df = pd.DataFrame(results)
    file_name = f"sentimentos_screenpipe_{nome}.csv"
    df.to_csv(file_name, index=False, encoding="utf-8-sig")

    print(f"\n📁 CSV gerado: {file_name}")
    print("📊 Distribuição de sentimentos:")
    print(df["Sentiment"].value_counts(normalize=True) * 100)
    print("-" * 50)

print("\n✅ Processo concluído com sucesso!")
