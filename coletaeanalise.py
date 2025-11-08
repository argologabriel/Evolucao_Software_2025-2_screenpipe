import requests
import pandas as pd
import torch
from transformers import pipeline
import os
import gc

REPO_OWNER = "mediar-ai"
REPO_NAME = "screenpipe"
GITHUB_TOKEN = "substituir token" 

CSV_INTERMEDIARIO = f"github_comments_{REPO_OWNER}_{REPO_NAME}.csv"
CSV_FINAL_ANALISE = "sentimentos_analisados.csv"

MODELOS = {
    "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
    "twitter-roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "multilingual-student": "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
}

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}
API_BASE_URL = "https://api.github.com"
DEVICE = 0 if torch.cuda.is_available() else -1

def get_last_100_closed_prs(owner, repo):

    print(f"Buscando PRs para {owner}/{repo}...")
    prs_url = f"{API_BASE_URL}/repos/{owner}/{repo}/pulls"
    params = {"state": "closed", "per_page": 100, "page": 1, "sort": "updated", "direction": "desc"}
    response = requests.get(prs_url, headers=HEADERS, params=params)
    if response.status_code != 200:
        print(f"Erro ao buscar PRs: {response.status_code} - {response.json().get('message')}")
        return []
    print(f"Encontrados {len(response.json())} PRs.")
    return response.json()

def get_pr_comments(pr):

    pr_number = pr['number']
    all_comments = []
    
    comments_url = pr['comments_url']
    response_issue = requests.get(comments_url, headers=HEADERS)
    if response_issue.status_code == 200:
        for comment in response_issue.json():
            user_login = comment['user']['login'] if comment['user'] else "usuario-deletado"
            all_comments.append({
                "pr_number": pr_number, "pr_title": pr['title'], "comment_id": comment['id'],
                "comment_body": comment['body'], "comment_user": user_login,
                "comment_type": "issue_comment", "created_at": comment['created_at']
            })

    review_comments_url = pr['review_comments_url']
    response_review = requests.get(review_comments_url, headers=HEADERS)
    if response_review.status_code == 200:
        for comment in response_review.json():
            user_login = comment['user']['login'] if comment['user'] else "usuario-deletado"
            all_comments.append({
                "pr_number": pr_number, "pr_title": pr['title'], "comment_id": comment['id'],
                "comment_body": comment['body'], "comment_user": user_login,
                "comment_type": "review_comment", "created_at": comment['created_at']
            })
    return all_comments

def load_model(model_name):

    print(f"A carregar modelo: {model_name}...")
    try:
        return pipeline("sentiment-analysis", model=model_name, device=DEVICE)
    except Exception as e:
        print(f"Erro ao carregar {model_name}: {e}")
        return None

def analyze_sentiment(text, classifier):
 
    try:
        if not isinstance(text, str):
            text = str(text)
        result = classifier(text, truncation=True, max_length=512)
        label = result[0]['label'].upper()
        score = result[0]['score']
        
        if label == "LABEL_2" or "POSITIVE" in label:
            label = "POSITIVE"
        elif label == "LABEL_0" or "NEGATIVE" in label:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        return label, score
    except Exception:
        return "ERROR", 0.0

def main():

    print("--- Passo 1: Coleta de Dados do GitHub ---")
    
    if GITHUB_TOKEN == "substituir token":
        print("Você esqueceu de substituir seu Token")
        return

    pull_requests = get_last_100_closed_prs(REPO_OWNER, REPO_NAME)
    if not pull_requests: 
        print("Nenhum PR encontrado ou erro na API.")
        return

    all_data = []
    total_comments = 0
    print(f"Iniciando coleta de comentários para {len(pull_requests)} PRs...")
    
    for i, pr in enumerate(pull_requests):
        print(f"  Coletando comentários do PR {i+1}/{len(pull_requests)}: {pr['title']}") 
        comments = get_pr_comments(pr)
        total_comments += len(comments)
        all_data.extend(comments)

    print(f"\nColeta concluída! Total de {total_comments} comentários encontrados.")
    
    df = pd.DataFrame(all_data)
    
    df = df.dropna(subset=['comment_body'])
    df = df[df['comment_body'].str.strip() != '']
    
    df.to_csv(CSV_INTERMEDIARIO, index=False, encoding='utf-8')
    print(f"Backup intermediário salvo em: {CSV_INTERMEDIARIO}")
    print("--- PASSO 1 CONCLUÍDO ---")
    
    print(f"\n--- INICIANDO PASSO 2: Análise de Sentimentos ---")
    
    for key, model_name in MODELOS.items():
         
        classifier = load_model(model_name)
        if classifier is None:
            print(f"O modelo {key} falhou.")
            continue
            
        print(f"\nModelo usado no momento: {key}")
        label_col = f"{key}_label"
        score_col = f"{key}_score"
        
        
        df[[label_col, score_col]] = df['comment_body'].apply(
            lambda text: pd.Series(analyze_sentiment(text, classifier))
        )
        print(f"Modelo {key} aplicado com sucesso.")
        
        del classifier
        gc.collect() 

    df.to_csv(CSV_FINAL_ANALISE, index=False, encoding='utf-8')
    print(f"\n--- PASSO 2 CONCLUÍDO ---")
    print(f"\nProcesso finalizado! O seu arquivo com as análises é: {CSV_FINAL_ANALISE}")

if __name__ == "__main__":
    main()