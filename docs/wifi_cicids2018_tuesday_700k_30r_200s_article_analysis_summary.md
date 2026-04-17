# Analise Automatizada: WiFi_CICIDS2018_Tuesday_700k_30r_200s_article

Resultados em: `/home/weverton/Artigo-LANC-Thiago/FL_NETWORK_FLOW_LLM/results/WiFi_CICIDS2018_Tuesday_700k_30r_200s_article`

## Melhor Resultado
- Rodada: 14
- K: 5
- Modo de threshold: fpr_target
- Threshold: 0.8750
- F1: 0.6276
- Precisao: 0.8358
- Recall: 0.5024
- Benign FPR: 0.0987

## Tamanho dos Splits
- Train benigno: 560000
- Teste total: 140000
- Teste benigno: 70000
- Teste anomalo: 70000
- Calibracao benigna: 35000

## Observacoes
- O detector ficou relativamente conservador, priorizando precisão acima de sensibilidade extrema.
- A melhor rodada apareceu no meio do treino, indicando aprendizado contínuo sem saturação imediata.

## Comparacao com Baseline
- Delta F1: -0.0686
- Delta Precisao: 0.2991
- Delta Recall: -0.4882
- Delta Benign FPR: -0.7564

## Rodadas por K
- K=1: melhor rodada 25 com F1=0.6260, Precision=0.8344, Recall=0.5010, BenignFPR=0.0994
- K=5: melhor rodada 14 com F1=0.6276, Precision=0.8358, Recall=0.5024, BenignFPR=0.0987

## Arquivos Detectados
- F1 rows: 60
- Temporal rows: 60
- Communication rows: 30
