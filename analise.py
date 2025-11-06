import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import os

MODELS = {
    'M_A_DistilBERT_SST2': 'distilbert-base-uncased-finetuned-sst-2-english', #
    
    'M_B_BERTweet': 'finiteautomata/bertweet-base-sentiment-analysis', 
    
    'M_C_RoBERTa_Latest': 'cardiffnlp/twitter-roberta-base-sentiment-latest' 
}
OUTPUT_FILENAME = 'pull_requests_comentarios.csv'

def map_sentiment(label, score, model_name):
    """Padroniza os rótulos de sentimento para 'POSITIVO', 'NEGATIVO' ou 'NEUTRO'."""
    label = label.upper()
    
    if model_name in [MODELS['M_B_BERTweet'], MODELS['M_C_RoBERTa_Latest']]:
        if 'NEG' in label or 'LABEL_0' in label: return 'NEGATIVO', score
        if 'NEU' in label or 'LABEL_1' in label: return 'NEUTRO', score
        if 'POS' in label or 'LABEL_2' in label: return 'POSITIVO', score

    else:
        if score < 0.6:
            return 'NEUTRO', 1.0 - score
        
        if 'POSITIVE' in label: return 'POSITIVO', score
        if 'NEGATIVE' in label: return 'NEGATIVO', score
    
    return 'NEUTRO', 0.5 # Default

def analyze_sentiment(df, model_key, model_name):
    """Executa a análise de sentimento para um modelo e adiciona os resultados ao DataFrame."""
    
    try:
        sentiment_pipeline = pipeline("text-classification", model=model_name, return_all_scores=True)
    except Exception as e:
        print(f"❌ ERRO GRAVE ao carregar o modelo {model_name}. Este modelo será pulado.")
        return

    results = []
    print(f"\nIniciando análise com o Modelo {model_key}...")

    for text in tqdm(df['comments_text'], desc=f"Analisando com {model_key}"):
        
        if not isinstance(text, str) or not text.strip():
            results.append({'sentiment': 'NEUTRO', 'score': 1.0})
            continue

        try:
            text_truncated = text[:500] 
            
            output_list = sentiment_pipeline(text_truncated)[0]
            best_output = max(output_list, key=lambda x: x['score'])
            
            sentiment, score = map_sentiment(best_output['label'], best_output['score'], model_name)
            results.append({'sentiment': sentiment, 'score': score})
            
        except Exception as e:
            results.append({'sentiment': 'ERRO', 'score': 0.0})

    df[f'sentimento_{model_key}'] = [r['sentiment'] for r in results]
    df[f'score_{model_key}'] = [r['score'] for r in results]

if __name__ == "__main__":
    
    try:
        df_pull_requests = pd.read_csv(OUTPUT_FILENAME, encoding='utf-8')
        print(f"\n✅ DataFrame '{OUTPUT_FILENAME}' carregado com sucesso.")
    except FileNotFoundError:
        print(f"\n❌ Erro: O arquivo '{OUTPUT_FILENAME}' não foi encontrado. Execute o 'extracao.py' primeiro.")
        exit()

    for key, name in MODELS.items():
        analyze_sentiment(df_pull_requests, key, name)
        
    print("\n\n--- 🚀 Análise de Sentimentos Concluída ---")

    final_output_filename = 'analise_sentimentos_final.csv'
    df_pull_requests.to_csv(final_output_filename, index=False, encoding='utf-8')
    print(f"✅ Resultados detalhados salvos em '{final_output_filename}'")

    
    print("\n### 📊 Resultados da Análise de Sentimentos (Resumido em %)")
    summary = {}
    
    for col_key in MODELS.keys():
        col_name = f'sentimento_{col_key}'
        
        if col_name in df_pull_requests.columns:
            summary[col_key] = (df_pull_requests[col_name].value_counts(normalize=True) * 100).round(2).to_dict()
        else:
            summary[col_key] = {"ANALISE FALHOU": 100.0}
    
    df_summary = pd.DataFrame(summary).fillna(0).sort_index()
    print(df_summary.to_markdown())

    comparison_cols = ['pr_number']
    for key in MODELS.keys():
        col_name = f'sentimento_{key}'
        if col_name in df_pull_requests.columns:
            comparison_cols.append(col_name)
    
    tabela_comparacao = df_pull_requests[comparison_cols]
    
    print("\n### 📋 Tabela de Comparação dos 100 Pull Requests (Primeiras 10 linhas)")
    print(tabela_comparacao.head(10).to_markdown(index=False))