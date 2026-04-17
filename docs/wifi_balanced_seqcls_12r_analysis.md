# Analise Tecnica da Metodologia Balanceada Supervisionada e dos Resultados 12r

## 1. Objetivo

Este documento consolida a metodologia ativa do experimento `WiFi_CICIDS2018_Tuesday_balanced_seqcls_1152k_12r_200s` e analisa seus resultados em comparacao com:

- o experimento supervisionado anterior de 30 rodadas: `WiFi_CICIDS2018_Tuesday_balanced_seqcls_1152k_30r_200s`
- a baseline anterior baseada na estrategia legacy: `WiFi_CICIDS2018_Tuesday_700k_30r_200s_article`

O foco aqui e duplo:

- documentar com clareza o que a nova metodologia faz
- interpretar tecnicamente os resultados do novo run de 12 rodadas

## 2. Metodologia Atual

### 2.1 Estrategia de tarefa

A configuracao atual esta em `configs/config_wifi.yaml` e define:

- `training_task: "sequence_classification"`
- `num_labels: 2`
- `anomaly_label_id: 1`

Isso significa que o sistema atual nao opera mais como uma deteccao puramente `one-class` baseada em modelagem causal de logs. Em vez disso, ele executa classificacao supervisionada binaria:

- classe `0`: benigno
- classe `1`: anomalo

### 2.2 Conversao fluxo -> texto

O pipeline ainda preserva a ideia central do projeto: representar fluxos de rede como sequencias textuais. O processador ativo e `src/data_processing/wifi_processor.py`.

As features principais usadas para compor o campo `Content` incluem:

- `Protocol`
- `Flow Duration`
- `Tot Fwd Pkts`
- `Tot Bwd Pkts`
- `TotLen Fwd Pkts`
- `TotLen Bwd Pkts`
- `Flow Byts/s`
- `Flow Pkts/s`
- `Flow IAT Mean`
- `Fwd IAT Mean`
- `Bwd IAT Mean`
- `Pkt Len Mean`
- `Pkt Len Var`
- `SYN Flag Cnt`
- `ACK Flag Cnt`
- `PSH Flag Cnt`

Assim, a mudanca de metodologia nao abandona a representacao textual. O que muda e a tarefa de aprendizado imposta ao modelo.

### 2.3 Estrategia de split

O split atual usa:

- `wifi_split_strategy: "balanced_binary"`
- `balanced_anchor: "minority"`
- `train_fraction_per_class: 0.8`

Com isso, o processador:

1. identifica fluxos benignos e anomalos no CSV bruto do CIC-IDS2018 Tuesday
2. reduz a classe benigna para casar com a classe anomala minoritaria
3. faz split estratificado por classe em treino e teste
4. reserva benignos excedentes para calibracao operacional

Numeros do experimento ativo:

| Item | Valor |
|---|---:|
| Anomalias disponiveis | 576.191 |
| Benignos casados | 576.191 |
| Total balanceado | 1.152.382 |
| Treino | 921.904 |
| Teste | 230.478 |
| Calibracao benigna | 35.000 |

### 2.4 Modelo e adaptacao parametrica

O backbone continua sendo:

- `HuggingFaceTB/SmolLM-135M`

O carregamento agora usa `AutoModelForSequenceClassification` com LoRA.

Configuracao ativa:

| Parametro | Valor |
|---|---:|
| `lora` | `true` |
| `lora_rank` | `8` |
| `lora_alpha_multiplier` | `2` |
| `lora_alpha` efetivo | `16` |
| `lora_dropout` | `0.05` |

No modo `sequence_classification`, a cabeca `score` e preservada em `modules_to_save`, permitindo ajustar tanto os adapters LoRA quanto a cabeca de classificacao.

### 2.5 Federated learning

Politica federada ativa:

| Parametro | Valor |
|---|---:|
| `num_rounds` | `12` |
| `num_clients` | `50` |
| `client_frac` | `0.2` |
| Clientes por rodada | `10` |
| `data_distribution_strategy` | `iid` |
| `client_selection_strategy` | `uniform` |
| `aggregation_method` | `FedAvg` |
| `fedprox_mu` | `0.0` |

Cada rodada seleciona 10 clientes entre 50 e agrega apenas os parametros treinaveis.

### 2.6 Politica de treino local

Configuracao local ativa:

| Parametro | Valor |
|---|---:|
| `batch_size` | `2` |
| `max_steps` | `200` |
| `num_train_epochs` | `1` |
| `initial_lr` | `0.001` |
| `lr_scheduler_type` | `constant` |
| `train_torch_dtype` | `float16` |
| `train_mixed_precision` | `fp16` |

O batch size permaneceu em `2`. Essa decisao foi intencional:

- o batch nao era a causa principal do colapso observado anteriormente
- manter o batch fixo reduz a quantidade de variaveis mudando ao mesmo tempo
- isso facilita atribuir a melhora observada as alteracoes realmente relevantes

### 2.7 Politica de avaliacao

Aqui ocorreu a mudanca mais importante para estabilizacao do experimento.

Configuracao atual:

| Parametro | Valor |
|---|---:|
| `eval_batch_size` | `8` |
| `eval_use_autocast` | `false` |
| `eval_torch_dtype` | `float32` |
| `threshold_selection` | `fpr_target` |
| `fpr_target` | `0.10` |
| `calibration_source` | `calibration_benign` |
| `calibration_num_samples` | `35000` |

No modo atual:

1. o score por fluxo e a probabilidade da classe anomala
2. o threshold e escolhido pelo quantil dos scores benignos de calibracao
3. a regra operacional e `score >= threshold`

### 2.8 Correcao de estabilidade numerica

No experimento antigo de 30 rodadas, a avaliacao em `fp16` fazia o threshold colapsar para `0.0` em rodadas tardias. Para corrigir isso, o novo run de 12 rodadas incorporou:

- avaliacao em `float32`
- `autocast` desligado na avaliacao
- selecao de threshold em `float64`
- salvaguarda: se o quantil benigno cair em `0.0`, mas ainda houver scores positivos, usa-se o menor valor positivo

Essas mudancas nao alteram a metodologia central do experimento. Elas corrigem um problema numerico da etapa de avaliacao.

## 3. Resultados do Novo Run de 12 Rodadas

Arquivos analisados:

- `results/WiFi_CICIDS2018_Tuesday_balanced_seqcls_1152k_12r_200s/f1_scores_fpr_target.csv`
- `results/WiFi_CICIDS2018_Tuesday_balanced_seqcls_1152k_12r_200s/temporal_metrics_fpr_target.csv`
- `results/WiFi_CICIDS2018_Tuesday_balanced_seqcls_1152k_12r_200s/communication_metrics.csv`

### 3.1 Desempenho geral

O novo run de 12 rodadas terminou com comportamento totalmente estavel.

Resumo global:

| Medida | Valor |
|---|---:|
| Melhor rodada | `8` |
| Melhor F1 | `0.9528` |
| Precision na melhor rodada | `0.9098` |
| Recall na melhor rodada | `1.0000` |
| Benign FPR na melhor rodada | `0.0991` |
| Threshold na melhor rodada | `2.82e-06` |

Ultima rodada:

| Medida | Valor |
|---|---:|
| Rodada final | `12` |
| F1 | `0.9511` |
| Precision | `0.9068` |
| Recall | `1.0000` |
| Benign FPR | `0.1027` |
| Threshold | `2.79e-07` |

Ponto importante: diferentemente do experimento de 30 rodadas, a rodada final continua valida e forte.

### 3.2 Estabilidade ao longo das rodadas

Todas as 12 rodadas foram estaveis:

- rodadas com `threshold > 0`: `12/12`
- rodadas colapsadas: `0`

Media do experimento:

| Medida | Valor |
|---|---:|
| F1 medio | `0.9510` |
| Desvio padrao do F1 | `0.00113` |
| Precision media | `0.9069` |
| Recall medio | `0.9996` |
| Benign FPR medio | `0.1027` |

Interpretacao:

- o modelo convergiu cedo
- a variacao entre rodadas foi muito pequena
- o regime final permaneceu operacionalmente consistente

### 3.3 Trajetoria das rodadas

Resumo por rodada:

| Rodada | F1 | Precision | Recall | Benign FPR | Threshold |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9510 | 0.9078 | 0.9986 | 0.1015 | 2.40e-03 |
| 2 | 0.9502 | 0.9063 | 0.9986 | 0.1033 | 4.67e-05 |
| 3 | 0.9509 | 0.9072 | 0.9990 | 0.1022 | 6.14e-06 |
| 4 | 0.9483 | 0.9018 | 0.9998 | 0.1089 | 3.43e-05 |
| 5 | 0.9500 | 0.9051 | 0.9996 | 0.1048 | 1.47e-05 |
| 6 | 0.9523 | 0.9089 | 1.0000 | 0.1002 | 1.15e-05 |
| 7 | 0.9504 | 0.9056 | 0.9998 | 0.1042 | 9.92e-05 |
| 8 | 0.9528 | 0.9098 | 1.0000 | 0.0991 | 2.82e-06 |
| 9 | 0.9517 | 0.9080 | 0.9999 | 0.1013 | 1.60e-06 |
| 10 | 0.9515 | 0.9075 | 1.0000 | 0.1019 | 1.65e-06 |
| 11 | 0.9514 | 0.9073 | 1.0000 | 0.1021 | 4.08e-06 |
| 12 | 0.9511 | 0.9068 | 1.0000 | 0.1027 | 2.79e-07 |

Leitura tecnica:

- o desempenho maximo apareceu na rodada 8
- o experimento estabilizou em uma faixa estreita muito cedo
- nao ha sinal de degradacao tardia estrutural dentro das 12 rodadas

## 4. Comparacao com o Experimento de 30 Rodadas da Mesma Metodologia

### 4.1 Melhor checkpoint

Comparacao entre os melhores checkpoints:

| Experimento | Melhor rodada | F1 | Precision | Recall | Benign FPR |
|---|---:|---:|---:|---:|---:|
| Novo `12r` | 8 | 0.9528 | 0.9098 | 1.0000 | 0.0991 |
| Antigo `30r` | 5 | 0.9523 | 0.9089 | 1.0000 | 0.1002 |

Diferenca `12r - 30r` no melhor checkpoint:

- `+0.00051` em F1
- `+0.00093` em precision
- `0.0` em recall
- `-0.00113` em benign FPR

Interpretacao:

- o novo `12r` nao apenas estabilizou o experimento
- ele tambem atingiu um melhor checkpoint ligeiramente superior

### 4.2 Estabilidade global

Comparacao resumida:

| Medida | Novo `12r` | Antigo `30r` |
|---|---:|---:|
| Rodadas totais | 12 | 30 |
| Rodadas estaveis | 12 | 18 |
| Rodadas colapsadas | 0 | 12 |
| F1 medio global | 0.9510 | 0.8353 |
| Benign FPR medio global | 0.1027 | 0.4660 |

Interpretacao:

- o `30r` tinha bom desempenho apenas em parte do treino
- o `12r` preservou o desempenho forte e eliminou a parte colapsada

### 4.3 Leitura metodologica

O experimento de 30 rodadas foi essencial para diagnosticar o problema:

- ele mostrou que o modelo aprendia bem
- mas expunha o colapso operacional do threshold em rodadas tardias

O novo experimento de 12 rodadas aproveita essa evidencia:

- interrompe o treino dentro da zona estavel
- corrige a avaliacao para evitar quantizacao numerica destrutiva
- produz um resultado final usavel como resultado principal

## 5. Comparacao com a Baseline Anterior

### 5.1 Melhor baseline antiga

A baseline antiga apresentou:

| Cenário | Rodada | k | F1 | Precision | Recall | Benign FPR |
|---|---:|---:|---:|---:|---:|---:|
| Melhor `k=1` | 30 | 1 | 0.6260 | 0.8315 | 0.5020 | 0.1017 |
| Melhor qualquer `k` | 30 | 5 | 0.6273 | 0.8310 | 0.5038 | 0.1024 |

### 5.2 Ganho do novo experimento

Comparando o melhor `12r` com a melhor baseline `k=1`:

- `+0.3268` em F1
- `+0.0783` em precision
- `+0.4980` em recall
- `-0.0026` em benign FPR

Comparando o melhor `12r` com a melhor baseline em qualquer `k`:

- `+0.3255` em F1
- `+0.0788` em precision
- `+0.4962` em recall
- `-0.0033` em benign FPR

Interpretacao:

- o ganho principal da nova metodologia aparece em recall
- o sistema atual praticamente nao perde anomalias
- e faz isso mantendo o controle de falsos positivos

## 6. Metricas Temporais

Resultados temporais do `12r`:

| Medida | Valor medio |
|---|---:|
| Mean TTD medio | 4349.47 s |
| Median TTD medio | 4089.92 s |
| Detection coverage media | 1.0 |
| Benign FPR medio | 0.1027 |

Melhor rodada temporal por `mean_ttd_seconds`:

| Rodada | Mean TTD | Coverage | Benign FPR |
|---|---:|---:|---:|
| 6 | 0.0 s | 1.0 | 0.1002 |

Interpretacao:

- a cobertura de deteccao foi total em todas as rodadas
- ha rodadas com deteccao imediata segundo a definicao temporal adotada
- o sistema atual preserva forte desempenho temporal junto com a estabilidade operacional

## 7. Custo de Comunicacao

### 7.1 Novo experimento de 12 rodadas

| Medida | Valor |
|---|---:|
| Bytes totais por rodada | 18.478.080 |
| Bytes medios por cliente | 1.847.808 |
| Parametros agregados por rodada | 4.619.520 |
| Bytes totais no experimento | 221.736.960 |

### 7.2 Comparacao com 30 rodadas

| Experimento | Bytes totais |
|---|---:|
| Novo `12r` | 221.736.960 |
| Antigo `30r` | 554.342.400 |

Reducao:

- o `12r` usa cerca de `40%` do custo total do `30r`
- isso corresponde a uma economia de aproximadamente `332,6 MB`

### 7.3 Comparacao com baseline antiga

| Experimento | Bytes totais por rodada | Bytes totais |
|---|---:|---:|
| Novo `12r` | 18.478.080 | 221.736.960 |
| Baseline antiga `30r` | 18.370.560 | 551.116.800 |

Interpretacao:

- a metodologia nova nao introduziu explosao de custo por rodada
- o custo total caiu fortemente porque o numero de rodadas foi reduzido

## 8. Principal Causa da Instabilidade Antiga

O experimento de 30 rodadas mostrou que o problema principal nao era incapacidade do modelo de aprender, mas sim a etapa de avaliacao:

- scores benignos muito pequenos
- avaliacao em `fp16`
- threshold por quantil sobre scores benignos

Isso levava a thresholds como:

- `5.960464e-08`
- `2.980232e-07`
- `4.768372e-07`
- e finalmente `0.0`

Quando o threshold virava `0.0`, a regra `score >= threshold` marcava tudo como anomalo, causando:

- `precision = 0.5`
- `recall = 1.0`
- `benign_fpr = 1.0`

O novo `12r` elimina esse comportamento observavel.

## 9. Conclusoes

### 9.1 Sobre a metodologia

A nova metodologia supervisionada balanceada e tecnicamente consistente porque:

- usa os dados anômalos disponiveis de forma direta
- preserva a representacao fluxo->texto do projeto
- adapta o backbone com LoRA em classificacao binaria
- usa calibracao operacional separada para threshold

### 9.2 Sobre os resultados

O experimento `12r` representa hoje o resultado mais forte e mais confiavel da linha supervisionada:

- melhor F1 observado: `0.9528`
- precision: `0.9098`
- recall: `1.0`
- benign FPR: `0.0991`

### 9.3 Sobre a comparacao com o `30r`

O `30r` foi importante como experimento diagnostico, mas o `12r` e superior como resultado final porque:

- evita o colapso tardio
- mantem desempenho de pico
- melhora levemente o melhor checkpoint
- entrega um resultado final estavel

### 9.4 Sobre a comparacao com a baseline antiga

O novo experimento supervisionado supera claramente a baseline anterior, principalmente por:

- enorme ganho de recall
- ganho expressivo de F1
- manutencao de `Benign FPR` controlado

## 10. Recomendacao para o Artigo

Para a redacao do artigo, a recomendacao mais forte e:

1. usar o experimento `12r` como resultado principal da estrategia supervisionada balanceada
2. usar o experimento `30r` como evidencia do problema de instabilidade tardia e da necessidade de estabilizacao
3. manter a baseline antiga como comparacao historica/metodologica

Formulação sugerida:

> A estrategia supervisionada balanceada baseada em `sequence classification` apresentou o melhor desempenho global, atingindo `F1 = 0.9528`, `precision = 0.9098`, `recall = 1.0` e `Benign FPR = 0.0991`. Em contraste com a versao anterior de 30 rodadas, o experimento reduzido para 12 rodadas manteve todas as rodadas em regime estavel, eliminando o colapso tardio observado anteriormente durante a calibracao do threshold.

