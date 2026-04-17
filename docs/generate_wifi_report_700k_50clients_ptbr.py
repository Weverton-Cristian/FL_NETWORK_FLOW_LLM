from __future__ import annotations

from datetime import date
from pathlib import Path
import html

from generate_wifi_report import (
    DOCS_DIR,
    ROOT,
    _environment_snapshot,
    _fmt_float,
    _html_table,
    _pre_block,
    _read_csv,
    _run,
    convert_html_to_pdf,
    html_to_text,
    write_simple_pdf,
)


CONFIG_PATH = ROOT / "configs" / "config_wifi.yaml"
CURRENT_RESULTS_DIR = ROOT / "results" / "WiFi_CICIDS2018_Tuesday_700k"
BASELINE_RESULTS_DIR = ROOT / "results" / "WiFi_CICIDS2018_Tuesday"

CURRENT_F1_PATH = CURRENT_RESULTS_DIR / "f1_scores.csv"
CURRENT_TEMPORAL_PATH = CURRENT_RESULTS_DIR / "temporal_metrics.csv"
CURRENT_COMM_PATH = CURRENT_RESULTS_DIR / "communication_metrics.csv"
CURRENT_CLIENT_META_PATH = CURRENT_RESULTS_DIR / "client_data" / "client_data_metadata.json"

BASELINE_F1_PATH = BASELINE_RESULTS_DIR / "f1_scores.csv"

HTML_PATH = DOCS_DIR / "Relatorio_Academico_WiFi_700k_50clientes.html"
PDF_PATH = DOCS_DIR / "Relatorio_Academico_WiFi_700k_50clientes.pdf"


def _paragraphs(items: list[str]) -> str:
    return "\n".join(f"<p>{item}</p>" for item in items)


def _best_rows_by_k(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row["k"]
        if key not in best or float(row["f1_score"]) > float(best[key]["f1_score"]):
            best[key] = row
    return [best[key] for key in sorted(best, key=lambda k: int(k))]


def _load_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _format_best_table(rows: list[dict[str, str]]) -> str:
    return _html_table(
        [
            {
                "k": row["k"],
                "round": row["round"],
                "threshold": _fmt_float(row["threshold"], 2),
                "f1_score": _fmt_float(row["f1_score"]),
                "precision": _fmt_float(row["precision"]),
                "recall": _fmt_float(row["recall"]),
                "benign_fpr": _fmt_float(row["benign_fpr"]),
            }
            for row in rows
        ],
        [
            ("k", "K"),
            ("round", "Melhor Rodada"),
            ("threshold", "Limiar"),
            ("f1_score", "F1"),
            ("precision", "Precisão"),
            ("recall", "Revocação"),
            ("benign_fpr", "FPR Benigna"),
        ],
    )


def _format_full_f1_table(rows: list[dict[str, str]]) -> str:
    return _html_table(
        [
            {
                "round": row["round"],
                "k": row["k"],
                "threshold": _fmt_float(row["threshold"], 2),
                "f1_score": _fmt_float(row["f1_score"]),
                "precision": _fmt_float(row["precision"]),
                "recall": _fmt_float(row["recall"]),
                "benign_fpr": _fmt_float(row["benign_fpr"]),
            }
            for row in rows
        ],
        [
            ("round", "Rodada"),
            ("k", "K"),
            ("threshold", "Limiar"),
            ("f1_score", "F1"),
            ("precision", "Precisão"),
            ("recall", "Revocação"),
            ("benign_fpr", "FPR Benigna"),
        ],
    )


def _format_temporal_table(rows: list[dict[str, str]]) -> str:
    return _html_table(
        [
            {
                "round": row["round"],
                "k": row["k"],
                "mean_ttd_seconds": _fmt_float(row["mean_ttd_seconds"]),
                "median_ttd_seconds": _fmt_float(row["median_ttd_seconds"]),
                "detection_coverage": _fmt_float(row["detection_coverage"]),
                "benign_fpr": _fmt_float(row["benign_fpr"]),
                "num_attacked_devices": row["num_attacked_devices"],
            }
            for row in rows
        ],
        [
            ("round", "Rodada"),
            ("k", "K"),
            ("mean_ttd_seconds", "TTD Médio (s)"),
            ("median_ttd_seconds", "TTD Mediano (s)"),
            ("detection_coverage", "Cobertura"),
            ("benign_fpr", "FPR Benigna"),
            ("num_attacked_devices", "Dispositivos Atacados"),
        ],
    )


def _format_comm_table(rows: list[dict[str, str]]) -> str:
    return _html_table(
        [
            {
                "round": row["round"],
                "num_selected_clients": row["num_selected_clients"],
                "bytes_total": row["bytes_total"],
                "bytes_mean_per_client": row["bytes_mean_per_client"],
                "params_total": row["params_total"],
                "aggregation_method": row["aggregation_method"],
            }
            for row in rows
        ],
        [
            ("round", "Rodada"),
            ("num_selected_clients", "Clientes Selecionados"),
            ("bytes_total", "Bytes Totais"),
            ("bytes_mean_per_client", "Bytes Médios/Cliente"),
            ("params_total", "Parâmetros Totais"),
            ("aggregation_method", "Agregação"),
        ],
    )


def _format_client_table(meta: dict[str, int]) -> str:
    rows = [
        {"client_id": client_id, "samples": samples}
        for client_id, samples in sorted(meta.items(), key=lambda item: int(item[0]))
    ]
    return _html_table(rows, [("client_id", "Cliente"), ("samples", "Amostras")])


def build_html() -> str:
    current_f1 = _read_csv(CURRENT_F1_PATH)
    current_temporal = _read_csv(CURRENT_TEMPORAL_PATH)
    current_comm = _read_csv(CURRENT_COMM_PATH)
    current_client_meta = {str(k): int(v) for k, v in _load_json(CURRENT_CLIENT_META_PATH).items()}
    baseline_f1 = _read_csv(BASELINE_F1_PATH)

    env = _environment_snapshot()
    config_text = CONFIG_PATH.read_text(encoding="utf-8")

    current_best_overall = max(current_f1, key=lambda row: float(row["f1_score"]))
    current_best_k1 = max((row for row in current_f1 if row["k"] == "1"), key=lambda row: float(row["f1_score"]))
    current_best_k5 = max((row for row in current_f1 if row["k"] == "5"), key=lambda row: float(row["f1_score"]))
    baseline_best_k1 = max((row for row in baseline_f1 if row["k"] == "1"), key=lambda row: float(row["f1_score"]))

    delta_f1 = float(current_best_k1["f1_score"]) - float(baseline_best_k1["f1_score"])
    delta_precision = float(current_best_k1["precision"]) - float(baseline_best_k1["precision"])
    delta_recall = float(current_best_k1["recall"]) - float(baseline_best_k1["recall"])
    delta_fpr = float(current_best_k1["benign_fpr"]) - float(baseline_best_k1["benign_fpr"])

    raw_size = _run(["du", "-sh", str(ROOT / "data" / "wifi" / "raw")])
    processed_size = _run(["du", "-sh", str(ROOT / "data" / "wifi" / "processed")])
    current_results_size = _run(["du", "-sh", str(CURRENT_RESULTS_DIR)])

    training_summary = [
        {
            "item": "Logs totais do experimento",
            "value": "700.000",
        },
        {
            "item": "Treino benigno",
            "value": "560.000",
        },
        {
            "item": "Teste benigno",
            "value": "70.000",
        },
        {
            "item": "Teste anômalo",
            "value": "70.000",
        },
        {
            "item": "Clientes totais",
            "value": "50",
        },
        {
            "item": "Fração de clientes por rodada",
            "value": "0,2",
        },
        {
            "item": "Clientes selecionados por rodada",
            "value": "10",
        },
        {
            "item": "Batches de clientes por rodada",
            "value": "5 batches de 2 clientes",
        },
        {
            "item": "Amostras de treino por cliente",
            "value": "11.200",
        },
        {
            "item": "Rodadas federadas",
            "value": "5",
        },
        {
            "item": "Passos máximos por cliente/rodada",
            "value": "20",
        },
        {
            "item": "Tempo total de treino observado",
            "value": "09m 27s",
        },
        {
            "item": "Tempo médio por rodada",
            "value": "01m 53s",
        },
        {
            "item": "Tempo típico de treino por cliente",
            "value": "~15,2s a ~15,7s",
        },
    ]

    comparison_rows = [
        {
            "scenario": "Experimento anterior (250k / 2 clientes)",
            "best_round": baseline_best_k1["round"],
            "f1": _fmt_float(baseline_best_k1["f1_score"]),
            "precision": _fmt_float(baseline_best_k1["precision"]),
            "recall": _fmt_float(baseline_best_k1["recall"]),
            "benign_fpr": _fmt_float(baseline_best_k1["benign_fpr"]),
        },
        {
            "scenario": "Experimento atual (700k / 50 clientes)",
            "best_round": current_best_k1["round"],
            "f1": _fmt_float(current_best_k1["f1_score"]),
            "precision": _fmt_float(current_best_k1["precision"]),
            "recall": _fmt_float(current_best_k1["recall"]),
            "benign_fpr": _fmt_float(current_best_k1["benign_fpr"]),
        },
        {
            "scenario": "Delta (atual - anterior)",
            "best_round": "-",
            "f1": _fmt_float(delta_f1),
            "precision": _fmt_float(delta_precision),
            "recall": _fmt_float(delta_recall),
            "benign_fpr": _fmt_float(delta_fpr),
        },
    ]

    file_inventory = """FL_NETWORK_FLOW_LLM/
├── configs/config_wifi.yaml
├── data/wifi/raw/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv
├── data/wifi/processed/train.csv
├── data/wifi/processed/test.csv
├── data/wifi/processed/tokenized/
├── results/WiFi_CICIDS2018_Tuesday/
├── results/WiFi_CICIDS2018_Tuesday_700k/
│   ├── client_data/client_data_metadata.json
│   ├── communication_metrics.csv
│   ├── f1_scores.csv
│   ├── temporal_metrics.csv
│   └── round_0 ... round_5/
├── src/data_processing/wifi_processor.py
├── src/federated_learning/client.py
├── src/federated_learning/server.py
├── src/evaluation/evaluator.py
└── docs/
    ├── generate_wifi_report.py
    ├── generate_wifi_report_academic_ptbr.py
    ├── generate_wifi_report_700k_50clients_ptbr.py
    └── Relatorio_Academico_WiFi_700k_50clientes.pdf"""

    today = date.today().strftime("%d/%m/%Y")

    html_doc = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <title>Relatório Acadêmico - Experimento 700k / 50 Clientes</title>
  <style>
    @page {{
      size: A4;
      margin: 1.8cm;
    }}
    body {{
      font-family: "Liberation Serif", "DejaVu Serif", serif;
      color: #111827;
      line-height: 1.55;
      font-size: 11pt;
    }}
    h1, h2, h3 {{
      color: #0f172a;
      margin-top: 1.15em;
      margin-bottom: 0.45em;
    }}
    h1 {{
      font-size: 22pt;
      border-bottom: 2px solid #1e293b;
      padding-bottom: 8px;
    }}
    h2 {{
      font-size: 16pt;
      border-bottom: 1px solid #cbd5e1;
      padding-bottom: 3px;
    }}
    h3 {{
      font-size: 13pt;
    }}
    p {{
      margin: 0.45em 0;
      text-align: justify;
    }}
    ul {{
      margin: 0.35em 0 0.7em 1.25em;
    }}
    li {{
      margin: 0.18em 0;
    }}
    code {{
      background: #f1f5f9;
      padding: 0.12em 0.25em;
      border-radius: 3px;
      font-family: "Liberation Mono", monospace;
      font-size: 10pt;
    }}
    pre {{
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      padding: 10px;
      white-space: pre-wrap;
      font-family: "Liberation Mono", monospace;
      font-size: 9.3pt;
      line-height: 1.35;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 0.6em 0 1em 0;
      font-size: 10pt;
    }}
    th, td {{
      border: 1px solid #cbd5e1;
      padding: 6px 8px;
      vertical-align: top;
    }}
    th {{
      background: #e2e8f0;
      text-align: left;
    }}
    .capa {{
      min-height: 720px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      text-align: center;
      page-break-after: always;
    }}
    .capa h1 {{
      border: none;
      font-size: 24pt;
      margin-top: 2.5cm;
    }}
    .capa .subtitle {{
      font-size: 14pt;
      line-height: 1.6;
    }}
    .capa .meta {{
      font-size: 12pt;
      line-height: 1.7;
      margin-bottom: 1.5cm;
    }}
    .ok {{
      background: #f0fdf4;
      border-left: 4px solid #16a34a;
      padding: 10px 12px;
      margin: 0.8em 0;
    }}
    .warn {{
      background: #fff7ed;
      border-left: 4px solid #ea580c;
      padding: 10px 12px;
      margin: 0.8em 0;
    }}
  </style>
</head>
<body>
  <div class="capa">
    <div></div>
    <div>
      <h1>Relatório Acadêmico Completo</h1>
      <div class="subtitle">
        Experimento de Aprendizado Federado com <code>700.000</code> logs<br />
        e <code>50</code> clientes sobre o dataset<br />
        <code>CIC-IDS2018 - Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv</code>
      </div>
    </div>
    <div class="meta">
      <div><strong>Projeto:</strong> <code>FL_NETWORK_FLOW_LLM</code></div>
      <div><strong>Experimento:</strong> <code>WiFi_CICIDS2018_Tuesday_700k</code></div>
      <div><strong>Data de geração:</strong> {today}</div>
      <div><strong>Raiz do projeto:</strong> <code>{html.escape(str(ROOT))}</code></div>
    </div>
  </div>

  <h2>Resumo</h2>
  {_paragraphs([
      "Este relatório documenta de forma aprofundada o experimento final configurado com 700.000 logs processados do dataset CIC-IDS2018 Tuesday e 50 clientes federados no repositório <code>FL_NETWORK_FLOW_LLM</code>. O objetivo foi consolidar uma configuração computacionalmente viável e metodologicamente mais realista do que os experimentos preliminares, preservando a lógica de detecção de anomalias baseada em aprendizado do comportamento benigno e avaliação posterior em conjunto misto.",
      "A configuração final utilizou 560.000 fluxos benignos para treino, 70.000 fluxos benignos para teste e 70.000 fluxos anômalos para teste, totalizando 700.000 logs. O cenário federado foi organizado em 50 clientes com divisão IID, participação de 20% dos clientes por rodada, 5 rodadas federadas, 10 clientes selecionados por rodada, 2 GPUs e 20 passos de treino por cliente selecionado em cada rodada.",
      f"O melhor resultado observado ocorreu em <code>K=1</code> na rodada <code>{current_best_k1['round']}</code>, com <code>F1 = {_fmt_float(current_best_k1['f1_score'])}</code>, <code>Precisão = {_fmt_float(current_best_k1['precision'])}</code>, <code>Revocação = {_fmt_float(current_best_k1['recall'])}</code> e <code>FPR Benigna = {_fmt_float(current_best_k1['benign_fpr'])}</code>. Em comparação com o experimento anterior de 250 mil logs e 2 clientes, houve leve ganho em F1 e pequena redução da FPR benigna, indicando melhoria incremental, embora a taxa de falso positivo ainda permaneça alta para uso operacional."
  ])}
  <p><strong>Palavras-chave:</strong> aprendizado federado, detecção de anomalias, CIC-IDS2018, modelos de linguagem, LoRA, análise de tráfego de rede, otimização de experimentos.</p>

  <h2>Abstract</h2>
  {_paragraphs([
      "This report presents a complete academic account of the final 700k-log / 50-client experiment conducted on top of the <code>FL_NETWORK_FLOW_LLM</code> repository using the CIC-IDS2018 Tuesday flow dataset. The experiment was designed to balance computational practicality and methodological realism after earlier smaller and much larger trial configurations.",
      f"The final setup used 560,000 benign training flows, 70,000 benign test flows, 70,000 anomalous test flows, 50 IID clients, 20% client participation per round, and 5 federated rounds. The best result was obtained at round {current_best_k1['round']} with top-1 F1 = {_fmt_float(current_best_k1['f1_score'])}. The new setup slightly improved F1 and benign false-positive rate over the previous 250k / 2-client experiment, while remaining much more operationally feasible than the full-dataset attempt."
  ])}

  <h2>1. Introdução</h2>
  {_paragraphs([
      "Após a integração bem-sucedida do dataset CIC-IDS2018 Tuesday ao repositório, surgiram três cenários experimentais distintos: um experimento reduzido com 250 mil logs e 2 clientes, uma tentativa de execução do dataset completo com treinamento por época que se mostrou excessivamente custosa, e por fim o experimento atual, concebido como compromisso entre fidelidade e viabilidade.",
      "O experimento com 700 mil logs e 50 clientes foi escolhido porque preserva diversidade de dados, introduz um cenário federado mais próximo de um ambiente multi-cliente real e, ao mesmo tempo, mantém tempo de execução compatível com ciclos iterativos de pesquisa. Além disso, ele permite analisar explicitamente o efeito de aumentar a quantidade de clientes sem levar o treinamento a uma escala operacionalmente inviável.",
      "Este relatório é mais completo do que a documentação anterior porque não apenas descreve as mudanças de código e os artefatos gerados, mas também situa o experimento atual dentro da trajetória de adaptação do projeto, compara quantitativamente os resultados com o baseline anterior e explicita as implicações metodológicas de cada decisão de configuração."
  ])}

  <h2>2. Contexto, Motivação e Histórico da Adaptação</h2>
  {_paragraphs([
      "A trilha de Wi-Fi do repositório não havia sido inicialmente projetada para processar um CSV massivo de fluxos de rede. A implementação original dependia de dois pequenos arquivos demonstrativos e de um conjunto simplificado de atributos. Isso foi adequado apenas enquanto o objetivo era demonstrativo.",
      "A primeira grande adaptação tornou o pipeline capaz de ingerir o arquivo <code>Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv</code> diretamente de <code>data/wifi/raw/</code>. Em seguida, foram introduzidos controles de cap para permitir experimentação progressiva. O experimento de 250 mil logs e 2 clientes validou a funcionalidade geral do pipeline, mas ainda representava um cenário pequeno. A tentativa subsequente com o dataset completo mostrou que treinar com todos os dados e todas as épocas tornaria o custo temporal excessivo.",
      "O cenário de 700 mil logs e 50 clientes nasce justamente dessa transição: ele mantém o pipeline real, usa o dataset original sem adulterá-lo, aumenta o número de clientes e amplia o volume de dados em relação ao primeiro experimento, mas faz isso dentro de um orçamento de tempo claramente mais controlado."
  ])}

  <h2>3. Dataset Utilizado e Estratégia de Armazenamento</h2>
  {_paragraphs([
      "O dataset utilizado é o arquivo <code>Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv</code>, pertencente à base CIC-IDS2018. O arquivo bruto foi mantido em <code>data/wifi/raw/</code>, enquanto todos os artefatos derivados ficaram em <code>data/wifi/processed/</code>. Essa separação preserva reprodutibilidade, evita alterações acidentais sobre o dado original e facilita refazer o experimento com diferentes configurações.",
      "A estrutura final do trabalho manteve o dado bruto como fonte imutável e tratou o conjunto processado como artefato descartável e recriável. Isso foi importante porque ao longo do projeto houve múltiplas iterações de configuração, incluindo o experimento anterior, a tentativa full dataset e o experimento 700k + 50 clientes."
  ])}
  <div class="ok">
    <strong>Dimensões observadas no servidor.</strong><br />
    Dados brutos: {html.escape(raw_size)}<br />
    Dados processados: {html.escape(processed_size)}<br />
    Resultados do experimento atual: {html.escape(current_results_size)}
  </div>

  <h2>4. Mudanças Estruturais no Código</h2>
  {_paragraphs([
      "O experimento atual reaproveita as mudanças estruturais já incorporadas ao projeto. Essas mudanças incluem a refatoração completa do processador de dados, a introdução de um arquivo de configuração específico para Wi-Fi/CIC-IDS2018, o uso de paralelismo controlado entre GPUs, a parametrização conservadora do DataLoader e a reescrita da avaliação para inferência em lotes.",
      "O arquivo <code>src/data_processing/wifi_processor.py</code> passou a ler um único CSV real, normalizar os rótulos, preservar metadados importantes e construir o campo <code>Content</code> somente depois da aplicação dos caps. Essa última alteração reduziu drasticamente trabalho desnecessário no preprocessamento.",
      "O arquivo <code>src/federated_learning/client.py</code> passou a aceitar tanto treinos por número fixo de passos quanto treinos dirigidos por época. Para o experimento atual, optou-se por <code>max_steps = 20</code> por cliente selecionado em cada rodada, o que limita o custo de execução.",
      "O arquivo <code>src/federated_learning/server.py</code> já suportava paralelismo, mas foi configurado e refinado para respeitar limites explícitos de GPU. Já o arquivo <code>src/evaluation/evaluator.py</code> passou a trabalhar em lotes, tornando a avaliação de 140 mil fluxos por rodada viável em tempo razoável."
  ])}

  <h2>5. Configuração Final do Experimento 700k + 50 Clientes</h2>
  {_paragraphs([
      "A configuração final procurou refletir três objetivos simultâneos: aumentar o tamanho do experimento, introduzir número maior de clientes e manter tempo de execução aceitável. Para isso, o conjunto processado foi limitado a 700 mil logs no total, preservando a lógica de treino benign-only do projeto.",
      "Os 560 mil logs de treino foram distribuídos IID entre 50 clientes, produzindo 11.200 exemplos benignos por cliente. Em cada rodada, 20% dos clientes são selecionados, o que resulta em 10 clientes participantes. Como o servidor dispõe de duas GPUs, esses 10 clientes são processados em 5 batches de 2 clientes."
  ])}
  {_html_table(training_summary, [("item", "Item"), ("value", "Valor")])}
  {_format_client_table(current_client_meta)}
  <div class="warn">
    <strong>Observação metodológica importante.</strong><br />
    Embora o experimento use 560 mil exemplos benignos no pool de treino, cada cliente selecionado executa apenas 20 passos por rodada. Portanto, o cenário atual não percorre integralmente todo o shard local em cada rodada; ele usa um orçamento de passos controlado para manter viabilidade computacional.
  </div>

  <h2>6. Como o Cenário com 50 Clientes Funciona na Prática</h2>
  {_paragraphs([
      "Com <code>num_clients = 50</code> e <code>client_frac = 0.2</code>, a seleção por rodada é calculada como <code>int(50 * 0.2) = 10</code>. Essa lógica decorre diretamente da implementação do servidor federado.",
      "Como o servidor possui apenas 2 GPUs, esses 10 clientes não são executados todos simultaneamente. Eles são quebrados em 5 batches sequenciais por rodada, cada batch contendo 2 clientes treinando em paralelo, um em cada GPU. Esse comportamento foi efetivamente observado nos logs, que mostraram sequências do tipo <code>Processing client batch 1/5</code> até <code>5/5</code>."
  ])}

  <h2>7. Execução Observada e Comportamento Temporal</h2>
  {_paragraphs([
      "O treino completo terminou em 09m27s, com média de 01m53s por rodada. Cada batch de clientes levou aproximadamente 15 segundos para concluir seus 20 passos, o que é coerente com o tempo global observado por rodada.",
      "Esse resultado é importante porque mostra que o aumento do número de clientes não inviabilizou o experimento. Ao contrário, com um orçamento controlado de passos por cliente, foi possível escalar de 2 para 50 clientes mantendo a execução em um intervalo de tempo muito razoável para pesquisa iterativa.",
      "Em seguida, a avaliação percorreu as 5 rodadas e processou o conjunto de teste completo de 140 mil fluxos em cada checkpoint. A fase de avaliação foi concluída com sucesso e gerou os arquivos <code>f1_scores.csv</code>, <code>temporal_metrics.csv</code> e <code>communication_metrics.csv</code>."
  ])}

  <h2>8. Resultados Quantitativos do Experimento Atual</h2>
  <h3>8.1 Melhores resultados por valor de K</h3>
  {_format_best_table(_best_rows_by_k(current_f1))}
  <h3>8.2 Resultados completos por rodada</h3>
  {_format_full_f1_table(current_f1)}
  <h3>8.3 Métricas temporais</h3>
  {_format_temporal_table(current_temporal)}
  <h3>8.4 Métricas de comunicação</h3>
  {_format_comm_table(current_comm)}

  <h2>9. Comparação com o Experimento Anterior (250k / 2 clientes)</h2>
  {_paragraphs([
      "Uma vantagem desta documentação é situar o experimento atual em relação ao baseline anteriormente executado com 250 mil logs e 2 clientes. Essa comparação é relevante porque mostra que o aumento do volume de dados e do número de clientes não apenas permaneceu viável, mas também trouxe uma pequena melhoria quantitativa nas métricas centrais.",
      f"No baseline anterior, o melhor ponto para <code>K=1</code> foi obtido na rodada {baseline_best_k1['round']}, com <code>F1 = {_fmt_float(baseline_best_k1['f1_score'])}</code> e <code>FPR Benigna = {_fmt_float(baseline_best_k1['benign_fpr'])}</code>. No experimento atual, o melhor ponto surgiu na rodada {current_best_k1['round']}, com <code>F1 = {_fmt_float(current_best_k1['f1_score'])}</code> e <code>FPR Benigna = {_fmt_float(current_best_k1['benign_fpr'])}</code>."
  ])}
  {_html_table(
      comparison_rows,
      [
          ("scenario", "Cenário"),
          ("best_round", "Melhor Rodada"),
          ("f1", "F1"),
          ("precision", "Precisão"),
          ("recall", "Revocação"),
          ("benign_fpr", "FPR Benigna"),
      ],
  )}
  <p>Em termos absolutos, o ganho foi modesto, mas real: <strong>delta de F1 = {_fmt_float(delta_f1)}</strong> e <strong>delta de FPR Benigna = {_fmt_float(delta_fpr)}</strong>. Como a FPR diminuiu, esse delta negativo é positivo do ponto de vista operacional.</p>

  <h2>10. Interpretação dos Resultados</h2>
  {_paragraphs([
      f"O melhor resultado do experimento atual ocorreu em <code>K=1</code> na rodada {current_best_k1['round']}, com <code>F1 = {_fmt_float(current_best_k1['f1_score'])}</code>. Isso indica que o modelo está efetivamente aprendendo um sinal útil de separação entre tráfego benigno e tráfego anômalo. A precisão ficou em torno de {_fmt_float(current_best_k1['precision'])}, enquanto a revocação permaneceu extremamente alta, em {_fmt_float(current_best_k1['recall'])}.",
      "Essa combinação revela um padrão já visto em experimentos anteriores: o modelo quase não perde anomalias, mas continua classificando benignos demais como suspeitos. Em outras palavras, o sistema se mostra sensível, porém ainda pouco específico.",
      "O fato de a melhor rodada ter surgido em torno da rodada 3, com estabilidade nas rodadas 4 e 5, sugere que o processo convergiu rapidamente. Isso reforça a escolha de manter 5 rodadas, já que aumentá-las substancialmente provavelmente geraria custo adicional maior do que o benefício incremental em desempenho."
  ])}

  <h2>11. Limitações e Considerações Metodológicas</h2>
  <ul>
    <li>O treino permanece benign-only, coerente com a proposta do projeto, mas isso limita comparações com abordagens supervisionadas.</li>
    <li>O uso de <code>max_steps = 20</code> por cliente/rodada reduz custo, mas também significa que cada cliente não percorre todo seu shard local a cada rodada.</li>
    <li>O limiar é selecionado com base em <code>f1_max</code>, estratégia útil para análise experimental, mas não necessariamente ideal para um detector operacional.</li>
    <li>Ainda que o conjunto total seja maior do que o baseline anterior, ele continua sendo um recorte controlado do dataset bruto.</li>
    <li>A FPR benigna continua alta, o que indica necessidade de calibração adicional antes de qualquer uso operacional.</li>
  </ul>

  <h2>12. Conclusão</h2>
  {_paragraphs([
      "O experimento com 700 mil logs e 50 clientes representa, até o momento, o melhor compromisso entre realismo federado, custo computacional e capacidade de experimentação no contexto deste projeto. Ele se mostrou mais completo e informativo do que o cenário anterior, sem herdar a inviabilidade temporal da tentativa full dataset.",
      "A execução foi estável, os resultados ficaram consistentes entre rodadas e houve leve melhora sobre o baseline anterior. O sistema continua apresentando alta sensibilidade e FPR benigna elevada, o que deixa claro que a próxima fronteira de melhoria não é apenas escalar o volume de dados, mas calibrar melhor o critério de decisão.",
      "Como base para documentação, esse experimento fornece um retrato mais fiel do estado atual do projeto: código adaptado para dados reais, pipeline federado funcional, execução multi-GPU controlada, cenário com 50 clientes e métricas suficientemente ricas para orientar os próximos passos científicos."
  ])}

  <h2>Apêndice A. Inventário de Arquivos</h2>
  {_pre_block(file_inventory)}

  <h2>Apêndice B. Configuração Completa do Experimento Atual</h2>
  {_pre_block(config_text)}

  <h2>Apêndice C. Comandos de Reprodução</h2>
  {_pre_block(
      "cd ~/Artigo-LANC-Thiago/FL_NETWORK_FLOW_LLM\\n"
      "source venv/bin/activate\\n"
      "rm -rf results/WiFi_CICIDS2018_Tuesday_700k\\n"
      "rm -rf data/wifi/processed/*\\n"
      "python main.py --config configs/config_wifi.yaml"
  )}

  <h2>Apêndice D. Evidências do Ambiente de Execução</h2>
  {_pre_block(env["hostname"] + "\\n" + env["uname"])}
  {_pre_block(env["lscpu"])}
  {_pre_block(env["free"])}

  <h2>Apêndice E. Localização dos Arquivos Gerados</h2>
  {_pre_block(
      f"HTML: {HTML_PATH}\\n"
      f"PDF: {PDF_PATH}\\n"
      f"Resultados atuais: {CURRENT_RESULTS_DIR}\\n"
      f"Resultados baseline: {BASELINE_RESULTS_DIR}\\n"
      f"Config: {CONFIG_PATH}"
  )}
</body>
</html>
"""
    return html_doc


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    html_doc = build_html()
    HTML_PATH.write_text(html_doc, encoding="utf-8")
    try:
        convert_html_to_pdf(HTML_PATH, PDF_PATH)
    except Exception:
        text_doc = html_to_text(html_doc)
        write_simple_pdf(text_doc, PDF_PATH)
    print(f"HTML report generated at: {HTML_PATH}")
    print(f"PDF report generated at: {PDF_PATH}")


if __name__ == "__main__":
    main()
