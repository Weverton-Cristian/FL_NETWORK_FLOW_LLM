from __future__ import annotations

import html
from datetime import date
from pathlib import Path

from generate_wifi_report import (
    CLIENT_META_PATH,
    COMM_PATH,
    CONFIG_PATH,
    DOCS_DIR,
    F1_PATH,
    ROOT,
    TEMPORAL_PATH,
    TRAINING_OBSERVATIONS,
    _best_rows_by_k,
    _environment_snapshot,
    _fmt_float,
    _html_table,
    _load_results,
    _pre_block,
    _read_csv,
    _run,
    convert_html_to_pdf,
    html_to_text,
    write_simple_pdf,
)


HTML_PATH = DOCS_DIR / "Relatorio_Academico_CICIDS2018_WiFi_PTBR.html"
PDF_PATH = DOCS_DIR / "Relatorio_Academico_CICIDS2018_WiFi_PTBR.pdf"


def _section_title(number: str, title: str) -> str:
    return f"{number}. {title}"


def _paragraphs(items: list[str]) -> str:
    return "\n".join(f"<p>{item}</p>" for item in items)


def _build_training_table() -> str:
    rows = [
        {
            "round": item["round"],
            "client_0_loss": f"{item['client_0_loss']:.4f}",
            "client_1_loss": f"{item['client_1_loss']:.4f}",
            "round_time_s": item["round_time_s"],
            "comment": item["comment"],
        }
        for item in TRAINING_OBSERVATIONS
    ]
    return _html_table(
        rows,
        [
            ("round", "Rodada"),
            ("client_0_loss", "Loss Cliente 0"),
            ("client_1_loss", "Loss Cliente 1"),
            ("round_time_s", "Tempo da Rodada (s)"),
            ("comment", "Interpretação"),
        ],
    )


def _build_f1_table(f1_rows: list[dict[str, str]]) -> str:
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
            for row in f1_rows
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


def _build_temporal_table(temporal_rows: list[dict[str, str]]) -> str:
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
            for row in temporal_rows
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


def _build_comm_table(comm_rows: list[dict[str, str]]) -> str:
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
            for row in comm_rows
        ],
        [
            ("round", "Rodada"),
            ("num_selected_clients", "Clientes Selecionados"),
            ("bytes_total", "Bytes Totais"),
            ("bytes_mean_per_client", "Bytes Médios/Cliente"),
            ("params_total", "Parâmetros"),
            ("aggregation_method", "Agregação"),
        ],
    )


def build_html() -> str:
    f1_rows, temporal_rows, comm_rows, client_meta = _load_results()
    env = _environment_snapshot()
    config_text = CONFIG_PATH.read_text(encoding="utf-8")

    best_by_k = _best_rows_by_k(f1_rows)
    best_overall = max(f1_rows, key=lambda row: float(row["f1_score"]))
    best_k1 = max((row for row in f1_rows if row["k"] == "1"), key=lambda row: float(row["f1_score"]))
    best_k5 = max((row for row in f1_rows if row["k"] == "5"), key=lambda row: float(row["f1_score"]))

    raw_size = _run(["du", "-sh", str(ROOT / "data" / "wifi" / "raw")])
    processed_size = _run(["du", "-sh", str(ROOT / "data" / "wifi" / "processed")])
    results_size = _run(["du", "-sh", str(ROOT / "results" / "WiFi_CICIDS2018_Tuesday")])

    file_inventory = """FL_NETWORK_FLOW_LLM/
├── configs/config_wifi.yaml
├── data/wifi/raw/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv
├── data/wifi/processed/train.csv
├── data/wifi/processed/test.csv
├── data/wifi/processed/tokenized/
├── results/WiFi_CICIDS2018_Tuesday/
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
    ├── WiFi_CICIDS2018_Technical_Report.pdf
    └── Relatorio_Academico_CICIDS2018_WiFi_PTBR.pdf"""

    toc = """Capa
Resumo
Abstract
1. Introdução
2. Objetivos
3. Caracterização do Repositório e do Cenário Inicial
4. Dataset Utilizado e Estratégia de Armazenamento
5. Metodologia de Adaptação do Pipeline
6. Alterações Implementadas no Código
7. Metodologia Experimental
8. Ambiente Computacional e Restrições do Servidor
9. Otimizações de Desempenho
10. Execução do Experimento Final
11. Resultados
12. Discussão
13. Limitações e Ameaças à Validade
14. Trabalhos Futuros
15. Conclusão
Apêndice A. Inventário de Arquivos
Apêndice B. Configuração Completa
Apêndice C. Comandos de Reprodução"""

    training_table = _build_training_table()
    f1_table = _build_f1_table(f1_rows)
    temporal_table = _build_temporal_table(temporal_rows)
    comm_table = _build_comm_table(comm_rows)
    best_by_k_table = _html_table(
        best_by_k,
        [
            ("k", "K"),
            ("round", "Melhor Rodada"),
            ("f1", "F1"),
            ("precision", "Precisão"),
            ("recall", "Revocação"),
            ("benign_fpr", "FPR Benigna"),
            ("threshold", "Limiar"),
        ],
    )
    client_table = _html_table(
        [{"client_id": k, "samples": v} for k, v in sorted(client_meta.items(), key=lambda item: int(item[0]))],
        [("client_id", "Cliente"), ("samples", "Amostras")],
    )

    today = date.today().strftime("%d/%m/%Y")

    html_doc = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <title>Relatório Acadêmico - CIC-IDS2018 Tuesday no FL_NETWORK_FLOW_LLM</title>
  <style>
    @page {{
      size: A4;
      margin: 1.8cm;
    }}
    body {{
      font-family: "Liberation Serif", "DejaVu Serif", serif;
      color: #111827;
      line-height: 1.5;
      font-size: 11pt;
    }}
    h1, h2, h3 {{
      color: #0f172a;
      margin-top: 1.1em;
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
      margin: 0.35em 0 0.65em 1.25em;
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
    .toc {{
      background: #f8fafc;
      border-left: 4px solid #2563eb;
      padding: 10px 12px;
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
    .page-break {{
      page-break-before: always;
    }}
  </style>
</head>
<body>
  <div class="capa">
    <div></div>
    <div>
      <h1>Relatório Acadêmico</h1>
      <div class="subtitle">
        Adaptação do Pipeline <code>FL_NETWORK_FLOW_LLM</code> para o Dataset<br />
        <code>CIC-IDS2018 - Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv</code>
      </div>
    </div>
    <div class="meta">
      <div><strong>Projeto:</strong> <code>FL_NETWORK_FLOW_LLM</code></div>
      <div><strong>Experimento:</strong> <code>WiFi_CICIDS2018_Tuesday</code></div>
      <div><strong>Data de geração:</strong> {today}</div>
      <div><strong>Raiz do projeto:</strong> <code>{html.escape(str(ROOT))}</code></div>
    </div>
  </div>

  <h2>Resumo</h2>
  {_paragraphs([
      "Este relatório documenta, em profundidade, o processo completo de adaptação do repositório <code>FL_NETWORK_FLOW_LLM</code> para uso com o dataset real <code>Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv</code>, pertencente à coleção CIC-IDS2018. A motivação central do trabalho foi substituir a antiga implementação de Wi-Fi, originalmente baseada em arquivos sintéticos e pequenos CSVs de demonstração, por um pipeline reproduzível capaz de consumir um grande arquivo de fluxos de rede, gerar conjuntos de treino e teste compatíveis com a abordagem de detecção de anomalias do projeto, executar treinamento federado com LoRA e avaliar os checkpoints produzidos.",
      "A adaptação envolveu alterações na configuração do experimento, refatoração completa do processador de dados, ajustes no servidor federado, inclusão de controles conservadores de DataLoader no cliente e reescrita da avaliação para inferência em lotes. Também foram realizadas decisões de engenharia voltadas ao ambiente de execução real, incluindo limites configuráveis para reduzir o tamanho efetivo do experimento sem modificar o dataset bruto e uso controlado de paralelismo em um servidor compartilhado.",
      "No experimento final, o arquivo bruto com aproximadamente 3,8 GB foi mantido em <code>data/wifi/raw/</code>, enquanto o conjunto processado foi reduzido para 250.000 fluxos benignos de treino e 250.000 fluxos de teste balanceados entre benignos e anômalos. O melhor resultado observado ocorreu na rodada 2 com <code>K=1</code>, alcançando <code>F1 = {_fmt_float(best_k1['f1_score'])}</code>, <code>Precisão = {_fmt_float(best_k1['precision'])}</code>, <code>Revocação = {_fmt_float(best_k1['recall'])}</code> e <code>FPR Benigna = {_fmt_float(best_k1['benign_fpr'])}</code>. Esses resultados mostram que o pipeline passou a funcionar adequadamente do ponto de vista técnico, ainda que a taxa de falsos positivos permaneça elevada para uso operacional imediato."
  ])}
  <p><strong>Palavras-chave:</strong> detecção de anomalias, aprendizado federado, LoRA, modelos de linguagem, CIC-IDS2018, fluxos de rede, otimização de desempenho.</p>

  <h2>Abstract</h2>
  {_paragraphs([
      "This report presents a comprehensive academic account of the end-to-end adaptation of the <code>FL_NETWORK_FLOW_LLM</code> repository to a real CIC-IDS2018 network-flow dataset. The work replaced a placeholder Wi-Fi pipeline with a reproducible large-scale CSV ingestion and processing path that supports benign-only training, balanced anomaly evaluation, federated LoRA fine-tuning, and batched checkpoint assessment.",
      "The implementation involved coordinated changes across configuration, data processing, server orchestration, client training, and evaluation. Special attention was given to performance and resource-awareness because the experiment ran on a shared server equipped with two RTX 4090 GPUs and 32 logical CPUs. The resulting pipeline is substantially more realistic, reproducible, and maintainable than the original placeholder implementation."
  ])}

  <h2>{_section_title("1", "Introdução")}</h2>
  {_paragraphs([
      "O repositório <code>FL_NETWORK_FLOW_LLM</code> foi concebido para explorar detecção de anomalias com apoio de modelos de linguagem em cenários de aprendizado federado. Entretanto, a trilha originalmente associada ao dataset de Wi-Fi não estava preparada para dados reais de larga escala. A implementação anterior dependia de dois arquivos pequenos, <code>benign_wifi.csv</code> e <code>anomaly_wifi.csv</code>, com um conjunto extremamente reduzido de atributos e até mesmo geração de dados sintéticos de contingência. Tal desenho era útil apenas como prova de conceito.",
      "A necessidade prática deste trabalho surgiu com a adoção do arquivo <code>Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv</code>, que possui 84 colunas e volume de aproximadamente 3,8 GB. Integrar esse dataset ao pipeline exigiu resolver múltiplos problemas: padronização de caminhos, leitura de um único CSV massivo, seleção de atributos relevantes, conversão do fluxo de rede em representação textual, separação entre treino benigno e teste misto, compatibilidade com o pipeline federado existente e avaliação em escala.",
      "Além do desafio funcional, havia uma restrição operacional relevante: a execução se dava em um servidor compartilhado por outros alunos. Dessa forma, a solução não poderia simplesmente maximizar o uso de hardware de maneira agressiva. Foi necessário buscar eficiência com responsabilidade, equilibrando tempo de execução, uso de CPU, uso de GPU, pressão de memória e reprodutibilidade experimental."
  ])}

  <h2>{_section_title("2", "Objetivos")}</h2>
  <ul>
    <li>Substituir a antiga implementação fictícia de Wi-Fi por um pipeline compatível com o CSV real do CIC-IDS2018.</li>
    <li>Organizar o dataset bruto e os artefatos processados segundo boas práticas de reprodutibilidade.</li>
    <li>Permitir experimentos controlados por meio de limites configuráveis, sem alterar o arquivo bruto original.</li>
    <li>Preservar a filosofia do projeto: treino apenas com tráfego benigno e detecção de anomalias na avaliação.</li>
    <li>Melhorar o desempenho do pipeline, sobretudo em preprocessamento e avaliação.</li>
    <li>Executar um experimento final documentado, com resultados e interpretação técnica.</li>
  </ul>

  <h2>{_section_title("3", "Caracterização do Repositório e do Cenário Inicial")}</h2>
  {_paragraphs([
      "Antes da refatoração, o caminho de processamento para o dataset de Wi-Fi não possuía aderência ao novo conjunto de dados. O código pressupunha duas fontes separadas, uma benigna e outra anômala, e trabalhava com um esquema pequeno de atributos como <code>rssi</code>, <code>snr</code>, <code>latency</code> e <code>throughput</code>. Isso inviabilizava o uso do CSV do CIC-IDS2018, que segue outra estrutura semântica e estatística.",
      "Outro problema do cenário inicial era a ausência de uma estratégia clara para limitar o tamanho do experimento sem editar o dataset original. Na prática, isso significava que, ao migrar para um arquivo de milhões de linhas, o custo de preprocessamento e avaliação tenderia a crescer rapidamente, dificultando iteração rápida.",
      "Também se observou que parte do pipeline subutilizava o servidor. A avaliação era majoritariamente serial, o paralelismo entre GPUs não estava ativado na configuração e a fase de treino não utilizava controles de DataLoader ajustados ao ambiente compartilhado."
  ])}

  <h2>{_section_title("4", "Dataset Utilizado e Estratégia de Armazenamento")}</h2>
  {_paragraphs([
      "O dataset integrado ao projeto foi o arquivo <code>Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv</code>, pertencente à base CIC-IDS2018 processada para algoritmos de aprendizado de máquina. O arquivo foi armazenado em <code>data/wifi/raw/</code>, preservando a política de manter dados brutos imutáveis. Os artefatos derivados, como <code>train.csv</code>, <code>test.csv</code> e o dataset tokenizado, passaram a ser gerados em <code>data/wifi/processed/</code>.",
      "Essa separação entre <code>raw</code> e <code>processed</code> é fundamental para rastreabilidade experimental. Ela impede que o arquivo original seja sobrescrito por transformações acidentais, reduz o risco de corrupção dos dados e permite repetir o experimento a partir da mesma fonte.",
      "Na validação inicial do arquivo foram confirmadas 84 colunas e a presença de campos cruciais para a nova lógica, como <code>Label</code>, <code>Timestamp</code>, <code>Src IP</code>, <code>Dst IP</code>, <code>Protocol</code>, <code>Flow Byts/s</code> e <code>Flow Duration</code>. Essa checagem foi importante para confirmar a compatibilidade estrutural do dataset com o novo processador."
  ])}
  <div class="ok">
    <strong>Dimensões finais dos artefatos.</strong><br />
    Dados brutos: {html.escape(raw_size)}<br />
    Dados processados: {html.escape(processed_size)}<br />
    Resultados: {html.escape(results_size)}
  </div>

  <h2>{_section_title("5", "Metodologia de Adaptação do Pipeline")}</h2>
  {_paragraphs([
      "A adaptação foi conduzida por etapas. Primeiro, definiu-se a estratégia de organização do dataset dentro do repositório. Em seguida, foi refeita a lógica de processamento para ler o CSV único, normalizar rótulos e gerar conjuntos derivados. Depois disso, ajustou-se o pipeline federado para operar de forma consistente com os novos artefatos. Por fim, foram introduzidas otimizações de desempenho visando reduzir tempo de execução e melhor aproveitar os recursos disponíveis.",
      "Um aspecto metodológico central foi preservar a semântica original do projeto: em vez de migrar para classificação supervisionada tradicional, manteve-se a lógica de detecção de anomalias via modelagem do comportamento benigno. Assim, o conjunto de treino final contém apenas fluxos benignos, enquanto o conjunto de teste mistura benignos e anômalos.",
      "Outro princípio importante foi evitar alterações destrutivas nos dados de origem. Todos os cortes de experimento, como limites máximos de amostras, foram implementados por configuração, nunca por edição manual do CSV bruto."
  ])}

  <h2>{_section_title("6", "Alterações Implementadas no Código")}</h2>
  <h3>6.1 <code>configs/config_wifi.yaml</code></h3>
  {_paragraphs([
      "Foi criado e refinado um arquivo de configuração específico para o experimento com o dataset CIC-IDS2018 Tuesday. Esse arquivo passou a declarar explicitamente o nome do CSV bruto, a coluna de rótulo, os valores considerados benignos, as colunas preferenciais usadas na montagem do texto, os limites do experimento, os hiperparâmetros do aprendizado federado e os parâmetros de desempenho.",
      "Entre os parâmetros mais importantes estão <code>benign_train_cap</code>, <code>test_benign_cap</code> e <code>test_anomaly_cap</code>, responsáveis por limitar o experimento a 250.000 amostras benignas de treino e 250.000 amostras totais de teste; <code>use_parallel_training</code> e <code>max_parallel_gpus</code>, que controlam o paralelismo entre GPUs; e <code>eval_batch_size</code>, <code>eval_use_autocast</code> e <code>eval_torch_dtype</code>, que otimizam a avaliação."
  ])}

  <h3>6.2 <code>src/data_processing/wifi_processor.py</code></h3>
  {_paragraphs([
      "O processador de Wi-Fi foi o componente mais profundamente alterado. A nova versão passou a ler um único CSV de grande porte, identificar a coluna de rótulo configurada, normalizar o campo <code>Label</code> para uma forma binária adequada ao pipeline, preservar a versão original do rótulo em <code>Label_raw</code> e manter metadados relevantes como <code>Flow ID</code>, <code>Src IP</code>, <code>Dst IP</code> e <code>Timestamp</code>.",
      "Também foi introduzida uma seleção flexível de atributos para a montagem do campo <code>Content</code>. Em vez de assumir cegamente um conjunto fixo de colunas, o código passa a escolher, dentre as colunas preferidas configuradas, apenas aquelas que realmente existem no CSV. Isso tornou o pipeline mais robusto a pequenas variações de esquema.",
      "Uma otimização particularmente importante foi mover a criação do campo textual <code>Content</code> para depois da aplicação dos limites do experimento. Antes disso, o código construía texto para um volume muito maior de linhas, inclusive para exemplos que seriam descartados mais tarde. Com a nova lógica, o texto só é gerado para o subconjunto efetivamente utilizado na execução."
  ])}
  <ul>
    <li>Suporte a um único CSV bruto real em <code>data/wifi/raw/</code>.</li>
    <li>Compatibilidade retroativa com os CSVs legados como fallback.</li>
    <li>Normalização robusta de rótulos benignos e anômalos.</li>
    <li>Preservação de metadados úteis para avaliação temporal.</li>
    <li>Aplicação de limites sem modificar o dataset bruto.</li>
    <li>Sanitização de padrões de IP e MAC no texto final.</li>
    <li>Geração de <code>train.csv</code>, <code>test.csv</code> e dataset tokenizado.</li>
  </ul>

  <h3>6.3 <code>src/federated_learning/server.py</code></h3>
  {_paragraphs([
      "O servidor federado já possuía um caminho de execução paralela, mas ele não estava sendo usado pelo experimento. A adaptação passou a ativar esse modo via configuração e, além disso, refinou o código para respeitar o parâmetro <code>max_parallel_gpus</code>. Isso permite limitar explicitamente quantas GPUs o servidor pode usar, o que é valioso em um ambiente compartilhado.",
      "Na prática, isso tornou possível usar as duas RTX 4090 disponíveis sem deixar a lógica dependente apenas da quantidade total de GPUs detectadas pela máquina."
  ])}

  <h3>6.4 <code>src/federated_learning/client.py</code></h3>
  {_paragraphs([
      "No cliente federado foram adicionados controles de DataLoader, com suporte a <code>dataloader_num_workers</code>, <code>dataloader_pin_memory</code> e <code>dataloader_persistent_workers</code>. A configuração final adotou quatro workers por processo, um valor conservador para melhorar a alimentação de dados sem pressionar excessivamente a CPU do servidor.",
      "A implementação foi feita de forma compatível com diferentes versões do <code>transformers</code>, evitando depender de parâmetros que possam não existir em instalações mais antigas."
  ])}

  <h3>6.5 <code>src/evaluation/evaluator.py</code></h3>
  {_paragraphs([
      "A avaliação recebeu a otimização mais impactante. O comportamento anterior processava uma amostra por vez, em um laço serial. Isso fazia com que a GPU fosse subutilizada e tornava a avaliação de centenas de milhares de fluxos muito lenta.",
      "A nova implementação passou a fazer tokenização e inferência em lotes, com suporte a autocast em CUDA e carregamento opcional do modelo em meia precisão. Além disso, o código passou a liberar explicitamente objetos de modelo e limpar cache CUDA entre rodadas. O resultado foi uma avaliação muito mais compatível com o volume do experimento."
  ])}

  <h2>{_section_title("7", "Metodologia Experimental")}</h2>
  {_paragraphs([
      "A metodologia final adotou um cenário controlado para equilibrar fidelidade experimental e custo computacional. Todos os fluxos benignos foram embaralhados, separados em parte de treino e holdout benigno, e então limitados por configuração. Os fluxos anômalos também foram limitados por configuração. O conjunto de treino final passou a conter apenas exemplos benignos, enquanto o conjunto de teste foi formado pela união do holdout benigno com a amostra anômala.",
      "O experimento utilizou dois clientes federados em distribuição IID, cinco rodadas de treinamento, modelo base <code>HuggingFaceTB/SmolLM-135M</code>, adaptação LoRA com rank 8, <code>batch_size = 2</code> e <code>max_steps = 20</code>. O treino foi conduzido com agregação <code>FedAvg</code> e sem termo FedProx adicional."
  ])}
  <ul>
    <li>Treino: 250.000 fluxos benignos.</li>
    <li>Teste: 125.000 fluxos benignos + 125.000 fluxos anômalos.</li>
    <li>Clientes: 2, com 125.000 amostras benignas para cada um.</li>
    <li>Rodadas federadas: 5.</li>
    <li>Valores de avaliação: <code>K = 1</code> e <code>K = 5</code>.</li>
  </ul>
  {client_table}

  <h2>{_section_title("8", "Ambiente Computacional e Restrições do Servidor")}</h2>
  {_paragraphs([
      "O experimento foi executado em um servidor acadêmico compartilhado. Isso impôs uma restrição metodológica importante: qualquer otimização precisava melhorar o desempenho sem provocar saturação desnecessária de CPU, GPU ou memória.",
      "O monitoramento realizado mostrou que, antes das otimizações, o processo utilizava efetivamente apenas um núcleo de CPU e subaproveitava a segunda GPU. Essas observações orientaram a decisão de ativar paralelismo conservador e melhorar a fase de avaliação."
  ])}
  <h3>8.1 Identificação do host</h3>
  {_pre_block(env["hostname"] + "\\n" + env["uname"])}
  <h3>8.2 CPU e memória</h3>
  {_pre_block(env["lscpu"])}
  {_pre_block(env["free"])}
  <h3>8.3 GPU</h3>
  {_pre_block(env["nvidia"])}

  <h2>{_section_title("9", "Otimizações de Desempenho")}</h2>
  {_paragraphs([
      "As otimizações implementadas foram guiadas por evidências coletadas durante a execução no servidor. Em vez de tentar escalar indiscriminadamente, optou-se por uma abordagem com bom custo-benefício para o contexto observado.",
      "A primeira otimização relevante foi adiar a geração do campo textual <code>Content</code> até depois da aplicação dos limites configurados. Isso reduziu trabalho desnecessário no preprocessamento. A segunda foi ativar o treino paralelo entre duas GPUs, respeitando um limite explícito. A terceira foi acrescentar workers de DataLoader em quantidade moderada. A quarta, e mais impactante, foi transformar a avaliação em um processo em lotes, com suporte a meia precisão."
  ])}
  <ul>
    <li>Paralelismo controlado entre duas GPUs.</li>
    <li>Quatro workers de DataLoader por processo.</li>
    <li><code>pin_memory</code> e <code>persistent_workers</code> habilitados.</li>
    <li>Avaliação com <code>eval_batch_size = 4</code>.</li>
    <li>Uso de <code>float16</code> e autocast na avaliação.</li>
    <li>Construção tardia do campo <code>Content</code>.</li>
  </ul>
  <div class="warn">
    <strong>Observação importante.</strong><br />
    No cenário final, o treino paralelo melhorou a utilização de hardware, mas não reduziu o tempo total de treino em relação ao cenário pequeno anterior, devido ao custo de inicialização de processos e carregamento duplicado do modelo. O maior ganho prático de velocidade veio da avaliação em lotes.
  </div>

  <h2>{_section_title("10", "Execução do Experimento Final")}</h2>
  {_paragraphs([
      "Após a limpeza dos artefatos derivados, o pipeline foi executado novamente desde o início com a configuração consolidada. O preprocessamento leu o CSV bruto, confirmou 16 colunas efetivamente usadas na construção do texto, aplicou os limites definidos e gerou <code>train.csv</code> e <code>test.csv</code> com as dimensões esperadas.",
      "A tokenização do conjunto de treino foi concluída e o servidor federado dividiu os dados entre dois clientes. O treinamento ocorreu em modo paralelo, utilizando as duas GPUs. Em seguida, a avaliação percorreu os checkpoints das cinco rodadas, gerando métricas de F1, precisão, revocação, FPR benigna e métricas temporais."
  ])}

  <h2>{_section_title("11", "Resultados")}</h2>
  <h3>11.1 Melhor checkpoint por configuração top-k</h3>
  {best_by_k_table}
  <p>O melhor resultado global foi obtido em <strong>rodada {html.escape(best_overall["round"])}</strong>, com <strong>K = {html.escape(best_overall["k"])}</strong> e <strong>F1 = {_fmt_float(best_overall["f1_score"])}</strong>.</p>

  <h3>11.2 Evolução do treinamento</h3>
  {training_table}

  <h3>11.3 Métricas de comunicação</h3>
  {comm_table}

  <h3>11.4 Tabela completa de F1</h3>
  {f1_table}

  <h3>11.5 Métricas temporais</h3>
  {temporal_table}

  <h2>{_section_title("12", "Discussão")}</h2>
  {_paragraphs([
      f"Os resultados mostram que a adaptação do pipeline foi bem-sucedida em termos de engenharia. O repositório passou a consumir um dataset real, de grande porte, e produziu artefatos consistentes de treino, teste, tokenização, treinamento federado e avaliação. Além disso, a curva de loss sugere aprendizagem rápida do padrão benigno, com queda acentuada entre as rodadas 1 e 2.",
      f"Do ponto de vista de desempenho preditivo, o melhor resultado em <code>K=1</code> ocorreu na rodada 2, com <code>F1 = {_fmt_float(best_k1['f1_score'])}</code>, <code>Precisão = {_fmt_float(best_k1['precision'])}</code> e <code>Revocação = {_fmt_float(best_k1['recall'])}</code>. Isso indica que o modelo realmente aprendeu um sinal de separação entre normalidade e anomalia. Entretanto, o valor de <code>FPR Benigna = {_fmt_float(best_k1['benign_fpr'])}</code> ainda é extremamente alto para um detector operacional.",
      f"Em <code>K=5</code>, o melhor resultado foi <code>F1 = {_fmt_float(best_k5['f1_score'])}</code>, inferior ao de <code>K=1</code>. Isso sugere que, para esta configuração específica de experimento, o top-1 foi mais informativo do que o top-5.",
      "As métricas temporais mostraram cobertura total e tempo de detecção nulo. Embora isso possa soar excelente, essa leitura precisa ser relativizada: um detector que acusa rapidamente quase tudo também pode manter alto nível de falsos positivos. Assim, o excelente TTD deve ser interpretado em conjunto com a FPR benigna, e não isoladamente."
  ])}

  <h2>{_section_title("13", "Limitações e Ameaças à Validade")}</h2>
  <ul>
    <li>O limiar foi escolhido em modo <code>f1_max</code>, o que é útil para análise exploratória, mas menos realista do que calibração operacional em conjunto separado.</li>
    <li>O experimento usou um subconjunto controlado do dataset, o que reduz custo, mas também limita representatividade.</li>
    <li>O treino continuou baseado apenas em tráfego benigno, o que é coerente com a proposta, mas pode restringir alternativas supervisionadas.</li>
    <li>O conjunto de avaliação foi balanceado artificialmente por limites configurados, o que simplifica comparação, porém não reflete necessariamente a prevalência real de anomalias em produção.</li>
    <li>O relatório interpreta o estado final do código e os logs observados; mudanças posteriores no repositório podem alterar parte das conclusões.</li>
  </ul>

  <h2>{_section_title("14", "Trabalhos Futuros")}</h2>
  <ul>
    <li>Criar um conjunto de calibração benigno e selecionar limiar por meta de FPR, não apenas por maximização de F1.</li>
    <li>Testar novas combinações de atributos no campo <code>Content</code>.</li>
    <li>Aumentar gradualmente <code>benign_train_cap</code> para verificar o efeito de mais dados de normalidade.</li>
    <li>Comparar distribuição IID com estratégias não IID entre clientes.</li>
    <li>Executar múltiplas seeds para avaliar estabilidade dos resultados.</li>
    <li>Investigar formas adicionais de redução de falsos positivos.</li>
  </ul>

  <h2>{_section_title("15", "Conclusão")}</h2>
  {_paragraphs([
      "Este trabalho transformou a trilha de Wi-Fi do repositório <code>FL_NETWORK_FLOW_LLM</code> de uma implementação demonstrativa para um pipeline funcional sobre dados reais de fluxo de rede. A integração do dataset CIC-IDS2018 Tuesday foi concluída com organização adequada dos dados, refatoração do processamento, ajustes no treinamento federado, otimizações de desempenho e avaliação consistente.",
      "O experimento final comprova que o sistema está operacional e que o modelo aprende rapidamente padrões de tráfego benigno. O melhor checkpoint surgiu cedo, na rodada 2, o que aponta convergência rápida sob o orçamento de treino adotado. Ainda assim, a taxa de falsos positivos sobre tráfego benigno permanece elevada, indicando que a base atual deve ser entendida como um baseline funcional e não como detector pronto para implantação.",
      "Do ponto de vista de engenharia, o ganho mais relevante foi tornar o repositório reproduzível, escalável para um dataset real e significativamente mais eficiente na fase de avaliação. Essa base agora permite ciclos futuros de experimentação mais rápidos, mais controlados e metodologicamente mais sólidos."
  ])}

  <h2>Apêndice A. Inventário de Arquivos</h2>
  {_pre_block(file_inventory)}

  <h2>Apêndice B. Configuração Completa do Experimento</h2>
  {_pre_block(config_text)}

  <h2>Apêndice C. Comandos de Reprodução</h2>
  {_pre_block(
      "cd ~/Artigo-LANC-Thiago/FL_NETWORK_FLOW_LLM\\n"
      "source venv/bin/activate\\n"
      "mkdir -p data/wifi/raw data/wifi/processed\\n"
      "wget -O data/wifi/raw/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv \\\"https://cse-cic-ids2018.s3.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv\\\"\\n"
      "python main.py --config configs/config_wifi.yaml"
  )}

  <h2>Apêndice D. Tabela de Conteúdo do Relatório</h2>
  <div class="toc">
    {_pre_block(toc)}
  </div>

  <h2>Apêndice E. Localização dos Arquivos Gerados</h2>
  {_pre_block(
      f"HTML: {HTML_PATH}\\n"
      f"PDF: {PDF_PATH}\\n"
      f"Config: {CONFIG_PATH}\\n"
      f"Resultados F1: {F1_PATH}\\n"
      f"Resultados Temporais: {TEMPORAL_PATH}\\n"
      f"Comunicação: {COMM_PATH}\\n"
      f"Metadados de clientes: {CLIENT_META_PATH}"
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
