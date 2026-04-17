# Analise Automatizada: WiFi_CICIDS2018_Tuesday_700k

Resultados em: `/home/weverton/Artigo-LANC-Thiago/FL_NETWORK_FLOW_LLM/results/WiFi_CICIDS2018_Tuesday_700k`

## Melhor Resultado
- Rodada: 3
- K: 1
- Modo de threshold: f1_max
- Threshold: 0.9000
- F1: 0.6962
- Precisao: 0.5367
- Recall: 0.9906
- Benign FPR: 0.8551

## Tamanho dos Splits
- Train benigno: 560000
- Teste total: 140000
- Teste benigno: 70000
- Teste anomalo: 70000
- Calibracao benigna: 35000

## Observacoes
- O detector está muito sensível: encontra quase todos os anômalos, mas ainda produz muitos falsos positivos benignos.
- A melhor rodada apareceu no meio do treino, indicando aprendizado contínuo sem saturação imediata.

## Rodadas por K
- K=1: melhor rodada 3 com F1=0.6962, Precision=0.5367, Recall=0.9906, BenignFPR=0.8551
- K=5: melhor rodada 5 com F1=0.6754, Precision=0.5156, Recall=0.9790, BenignFPR=0.9198

## Arquivos Detectados
- F1 rows: 10
- Temporal rows: 10
- Communication rows: 5
