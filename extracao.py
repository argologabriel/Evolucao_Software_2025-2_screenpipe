import requests
import pandas as pd
from tqdm import tqdm
import os


REPO_OWNER = "mediar-ai"
REPO_NAME = "screenpipe"
GITHUB_TOKEN = "" 

def fetch_pull_request_data():
    """Busca os 100 Pull Requests fechados mais recentes e seus comentários."""
    
    if GITHUB_TOKEN == "SEU_PAT_TOKEN_AQUI" or not GITHUB_TOKEN:
        print("ERRO: Configure o GITHUB_TOKEN com seu Token de Acesso Pessoal.")
        return pd.DataFrame()

    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    pulls_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls"
    params = {
        "state": "closed",
        "sort": "updated",
        "direction": "desc",
        "per_page": 100
    }
    
    response = requests.get(pulls_url, headers=headers, params=params)
    
    if response.status_code != 200:
        print(f"Erro ao buscar PRs (Status {response.status_code}): {response.text}")
        return pd.DataFrame()

    pull_requests = response.json()
    pr_data = []

    print(f"Buscando comentários para {len(pull_requests)} Pull Requests...")

    for pr in tqdm(pull_requests, desc="Processando PRs"):
        pr_number = pr['number']
        pr_title = pr['title']
        
        comments_url = pr['comments_url'] 
        
        comments_response = requests.get(comments_url, headers=headers)
        
        if comments_response.status_code == 200:
            comments = comments_response.json()
            
            all_comments_text = " ".join([c['body'] for c in comments if c.get('body')])
            
            pr_data.append({
                'pr_number': pr_number,
                'title': pr_title,
                'comments_text': all_comments_text,
                'num_comments': len(comments)
            })
        else:
            print(f"Erro ao buscar comentários do PR #{pr_number}. Código: {comments_response.status_code}")
            
    return pd.DataFrame(pr_data)

if __name__ == "__main__":
    df_pull_requests = fetch_pull_request_data()

    if not df_pull_requests.empty:
        output_filename = 'pull_requests_comentarios.csv'
        
        try:
            df_pull_requests.to_csv(output_filename, index=False, encoding='utf-8')
            print(f"\n✅ Dados da Fase 1 salvos com sucesso em '{output_filename}' ({len(df_pull_requests)} linhas).")
        except Exception as e:
            print(f"\n❌ Erro ao salvar o arquivo CSV: {e}")
    else:
        print("\n❌ A Fase 1 não gerou dados para salvar.")