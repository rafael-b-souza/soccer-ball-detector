# Soccer Ball Detector: Pipeline com YOLO11, Pré-Processamento Adaptativo e Rastreamento Otimizado por Algoritmo Genético (DEAP)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF.svg)](https://docs.ultralytics.com/)
[![DEAP 1.4+](https://img.shields.io/badge/DEAP-Genetic_Algorithm-FF6F00.svg)](https://deap.readthedocs.io/)
[![Paper](https://img.shields.io/badge/Paper-PDF_Acadêmico-red.svg)](docs/deteccao_de_bola_em_partidas_de_futebol.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI Tests](https://img.shields.io/badge/Tests-20_Passed-brightgreen.svg)](tests/)

Repositório oficial do trabalho acadêmico:  
**"Detecção de Bola em Partidas de Futebol: Um Pipeline com YOLO e Otimização via Algoritmo Genético"**  
Autores: **Rafael Bernary de Souza** e **Gabriel de Oliveira Rêgo** (Fevereiro de 2025).  
Documento acadêmico de referência: [`docs/deteccao_de_bola_em_partidas_de_futebol.pdf`](docs/deteccao_de_bola_em_partidas_de_futebol.pdf).

---

## ⚽ Demonstração Visual do Pipeline

O pipeline combina visão computacional em tempo real e computação evolutiva para rastrear a bola com precisão, isolando regiões dinâmicas e mitigando oclusões, desfoque de movimento (*motion blur*) e sombras do gramado.

### 1. Rastreamento Dinâmico em Jogo (YOLO11 + ROI + Histórico de Trajetória)

Demonstração do dashboard analítico operando com o rastreador de duplo estágio (*Dynamic ROI Tracking + Full-Frame Fallback*):

![Demonstração de Rastreamento em Tempo Real](docs/assets/tracking_demo.gif)

*Painel integrado: (Esquerda) Transmissão com detecção da bola, rastro vetorial neon da trajetória e janela da ROI dinâmica; (Superior Direito) Quadro após pré-processamento adaptativo ótimo ($\gamma = 1.21$, máscara HSV); (Inferior Direito) Recorte ampliado com zoom focal na ROI dinâmica ($4.27\times$).*

---

### 2. Estágios do Pré-Processamento Adaptativo de Imagem

Comparação visual lado a lado dos quatro estágios de refinamento aplicados ao quadro antes e durante a inferência da rede neural convolucional:

![Comparação dos Estágios do Pipeline](docs/assets/pipeline_comparison.png)

| Estágio | Operação | Objetivo Técnico e de Negócio |
| :--- | :--- | :--- |
| **Estágio 1: Raw Field Frame** | Imagem original do broadcast (BGR) | Entrada direta com variações de iluminação, refletores e sombras. |
| **Estágio 2: Correção Gamma Adaptativa** | Correção não linear via Look-Up Table ($\gamma = 1.211$) | Atenua sombras intensas e equaliza a intensidade luminosa da bola. |
| **Estágio 3: Segmentação de Cores HSV** | Filtro de fundo com máscara binária ($H \in [106, 156]$) | Suprime texturas repetitivas do gramado e isola elementos refletores. |
| **Estágio 4: YOLO11 + ROI + Trajetória** | Janela dinâmica ($4.27\times$) e rastro vetorial | Restringe o espaço de busca, eleva o FPS e assegura rastreabilidade contínua. |

---

### 3. Análise da Trajetória 2D e Dinâmica de Velocidade

Reconstrução da curva balística da bola na grande área e perfil de velocidade inter-quadros gerados automaticamente pelo sistema de telemetria:

![Análise da Trajetória 2D e Velocidade](docs/assets/trajectory_analysis.png)

---

## 🧬 Formulação Matemática do Algoritmo Genético (DEAP)

Para superar a necessidade de ajuste manual e empírico de parâmetros de visão computacional, o framework **DEAP** (*Distributed Evolutionary Algorithms in Python*) foi empregado na busca de hiperparâmetros de pré-processamento e rastreamento.

### 1. Função de Aptidão (Fitness Function)

A função objetivo formal formulada na Seção 3.3 do artigo pondera a acurácia de detecção ($\text{mAP}_{50}$) e a viabilidade de inferência em tempo real ($\text{FPS}$ normalizado):

$$\text{Fitness}(\mathbf{x}) = w_1 \cdot \text{mAP}_{50}(\mathbf{x}) + w_2 \cdot \left( \frac{\text{FPS}(\mathbf{x})}{\text{FPS}_{\text{base}}} \right)$$

Onde:
- $\mathbf{x}$: Vetor de cromossomos do indivíduo (configuração de hiperparâmetros).
- $\text{mAP}_{50}(\mathbf{x}) \in [0.0, 1.0]$: *Mean Average Precision* com limiar de interseção sobre união ($\text{IoU}$) igual a $0.50$ aferida no conjunto de validação.
- $\text{FPS}(\mathbf{x})$: Taxa média de quadros por segundo processados durante a inferência.
- $\text{FPS}_{\text{base}} = 30.0$: Fator de normalização que expressa a taxa padrão de captura de vídeos esportivos em tempo real ($30 \text{ quadros/s}$).
- $w_1 = 0.90$ ($90\%$ de peso atribuído à precisão e recall da detecção).
- $w_2 = 0.10$ ($10\%$ de peso atribuído ao throughput computacional).

### 2. Estrutura do Cromossomo (Espaço de Busca)

Cada indivíduo $\mathbf{x} = [x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7]$ representa um vetor cromossômico de 8 genes:

| Índice | Gene / Parâmetro | Tipo | Domínio / Faixa | Papel no Pipeline |
| :---: | :--- | :---: | :---: | :--- |
| $x_0$ | `preproc` | Categórico | $\{\text{none}, \text{hist\_eq}, \text{clahe}, \text{gamma}\}$ | Técnica principal de realce fotométrico. |
| $x_1$ | `remove_bg` | Booleano | $\{0, 1\}$ | Ativação da máscara de supressão de fundo. |
| $x_2$ | `lower_blue` | Inteiro | $[90, 110]$ | Limite inferior da matiz (Hue) no espaço HSV. |
| $x_3$ | `upper_blue` | Inteiro | $[130, 160]$ | Limite superior da matiz (Hue) no espaço HSV. |
| $x_4$ | `clahe_clip` | Contínuo | $[1.0, 4.0]$ | Limite de corte de contraste local no CLAHE. |
| $x_5$ | `tile_size` | Tupla $(N, M)$ | $\{4, 8, 16\} \times \{4, 8, 16\}$ | Tamanho da grade de equalização adaptativa. |
| $x_6$ | `gamma_value` | Contínuo | $[0.8, 1.5]$ | Fator $\gamma$ da curva $I_{\text{out}} = 255 \cdot (I_{\text{in}} / 255)^{1/\gamma}$. |
| $x_7$ | `roi_scale` | Contínuo | $[2.0, 5.0]$ | Fator multiplicador da janela ROI em relação à bola. |

### 3. Operadores Evolutivos e Estratégia de Convergência

- **Seleção**: Seleção por Torneio (*Tournament Selection*) com tamanho de torneio $k = 3$.
- **Cruzamento (Crossover)**: Crossover de Dois Pontos (*Two-Point Crossover*, $P_{\text{cx}} = 0.50$) com algoritmo corretor `safe_crossover` para reparo de integridade no gene discreto `tile_size`.
- **Mutação**: Mutação estocástica gênica uniforme ($P_{\text{mut}} = 0.20$), perturbando cada gene dentro dos limites de seu domínio estrito.
- **Protocolo Experimental**: População de 10 indivíduos submetida a 10 gerações evolutivas (totalizando 100 treinamentos rápidos de 10 épocas com GPU NVIDIA A100 no Google Colab).

### 4. Solução Ótima Selecionada vs. Amostras da População

Resultados empíricos extraídos do log de convergência [`data/ga_detailed_evaluations.csv`](data/ga_detailed_evaluations.csv):

| Configuração | Pré-Processamento | $\text{remove\_bg}$ | $\gamma$ | CLAHE Clip | Escala ROI | $\text{mAP}_{50}$ | FPS | Fitness Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ótimo (best_preproc_params.json)** | **gamma** | **True** | **1.211** | **2.838** | **4.273** | **0.4697** | **61.81** | **0.6287** |
| Candidato B (Geração 1) | gamma | False | 0.823 | 1.871 | 4.944 | 0.4697 | 59.31 | 0.6204 |
| Candidato C (Geração 2) | none | False | 1.275 | 3.520 | 3.836 | 0.4697 | 60.91 | 0.6257 |
| Candidato D (Geração 3) | clahe | True | 1.211 | 3.892 | 3.335 | 0.4697 | 60.88 | 0.6256 |
| Candidato E (Geração 5) | gamma | True | 1.110 | 3.739 | 3.564 | 0.4697 | 60.44 | 0.6242 |

---

## 📊 Resultados Experimentais Consolidados (Paper vs Baseline)

O treinamento final de 50 épocas na GPU NVIDIA L4 com os parâmetros ótimos do AG elevou substancialmente a precisão e a estabilidade da detecção. 

Em teste exaustivo de 1 minuto em partida de futebol real (Tabela 3 do artigo):

| Modelo e Configuração | Pré-Processamento | ROI Dinâmica | Detecções Confirmadas | FPS Médio | Ganho Relativo |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLO11 Final (Otimizado por AG)** | **Original** | **Não** | **1533** | **25.66** | **+111.7%** 🚀 |
| YOLO11 Final (Otimizado por AG) | Completo | Sim | 782 | 12.70 | Falsos positivos suprimidos |
| YOLO11 Final (Otimizado por AG) | Completo | Não | 760 | 18.09 | Filtragem fotométrica estrita |
| YOLO11 Small (Baseline Inicial) | Original | Não | 724 | 21.48 | Baseline de partida |
| YOLO11 Small (Baseline Inicial) | Completo | Não | 1406* | 17.81 | *Elevado índice de falsos positivos |
| YOLO11 Small (Baseline Inicial) | Completo | Sim | 1582* | 15.42 | *Elevado índice de falsos positivos |

> **Achado Científico Chave**: O modelo YOLO11 treinado com as transformações fotométricas ótimas superou o baseline em mais do dobro de detecções legítimas da bola (de 724 para 1533), eliminando oscilações espaciais e mantendo alto frame-rate.

---

## 📋 Checklist de Reprodutibilidade Científica

Tabela direta conectando cada hipótese, método e métrica do artigo acadêmico à implementação no repositório:

| Item do Paper | Seção no Artigo | Parâmetro / Métrica | Arquivo de Implementação | Como Verificar / Reproduzir |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset de Trechos** | Seção 3.1, Tab. 1 | 241 imagens (192 train, 49 val), 1 classe (`bola`) | `notebooks/pipeline_deteccao_bola.ipynb` | Extração Roboflow 3 FPS |
| **Pipeline de Imagem** | Seção 3.2, List. 1 | Gamma, CLAHE (LAB), Equalização (YUV), HSV | `src/soccer_ball_detector/preproc.py` | `pytest tests/test_preproc.py` |
| **Janela Dinâmica ROI** | Seção 3.2 | $ROI_{\text{scale}} \approx 4.27\times$, fallback automático | `src/soccer_ball_detector/roi.py` | `pytest tests/test_roi.py` |
| **Função de Aptidão** | Seção 3.3 | $\text{Score} = 0.90 \cdot \text{mAP} + 0.10 \cdot (\text{FPS}/30)$ | `src/soccer_ball_detector/optimizer.py` | `pytest tests/test_optimizer.py` |
| **Evolução via DEAP** | Seção 3.3 | 10 gerações, 10 indivíduos, torneio $k=3$ | `src/soccer_ball_detector/optimizer.py` | `GeneticOptimizer.run(...)` |
| **Log de Convergência** | Seção 4.1, Tab. 2 | 100 avaliações do AG com mAP50 e FPS | `data/ga_detailed_evaluations.csv` | `pd.read_csv('data/ga_detailed_evaluations.csv')` |
| **Configuração Ótima** | Seção 4.1 | $\gamma = 1.211$, remove_bg=True, scale=4.27 | `configs/best_preproc_params.json` | Carregado automaticamente no tracker |
| **Rastreamento e Fallback**| Seção 5, Discussão | Estágio duplo: ROI se alvo recente, senão Full | `src/soccer_ball_detector/detector.py` | `pytest tests/test_detector.py` |
| **Demonstração Visual** | Seção 4.3 | GIF de tracking e gráficos comparativos | `scripts/generate_demo_visuals.py` | `python scripts/generate_demo_visuals.py` |

---

## 🛠️ Arquitetura do Repositório

```text
soccer-ball-detector/
├── configs/
│   └── best_preproc_params.json         # Parâmetros ótimos da convergência do AG
├── data/
│   └── ga_detailed_evaluations.csv      # Log completo dos 100 experimentos evolutivos
├── docs/
│   ├── deteccao_de_bola_em_partidas_de_futebol.pdf # Artigo científico original
│   └── assets/                          # Recursos visuais para documentação
│       ├── tracking_demo.gif            # Demonstração animada do rastreador em jogo
│       ├── pipeline_comparison.png      # Comparação lado a lado dos 4 estágios fotométricos
│       └── trajectory_analysis.png      # Curva balística 2D e telemetria de velocidade
├── notebooks/
│   └── pipeline_deteccao_bola.ipynb     # Pipeline completo de ponta a ponta (Colab)
├── scripts/
│   └── generate_demo_visuals.py         # Gerador determinístico de assets de demonstração
├── src/
│   └── soccer_ball_detector/
│       ├── __init__.py                  # Exportação da interface pública da biblioteca
│       ├── detector.py                  # Tracker integrado YOLO11 com ROI dinâmica e fallback
│       ├── optimizer.py                 # Otimizador de hiperparâmetros com DEAP e fitness
│       ├── preproc.py                   # Transformações CLAHE, correção gamma e máscara HSV
│       ├── roi.py                       # Cálculo de coordenadas e recorte de ROI
│       ├── video.py                     # Pipeline de processamento e anotação em vídeo MP4
│       └── visuals.py                   # Simulador de partida sintética e compositor visual
├── tests/
│   ├── test_detector.py                 # Testes do SoccerBallTracker com PyTorch e Ultralytics
│   ├── test_optimizer.py                # Testes de operadores genéticos e convergência DEAP
│   ├── test_preproc.py                  # Testes das transformações e integridade dimensional
│   ├── test_roi.py                      # Testes de limites e extração de bboxes
│   └── test_visuals.py                  # Testes do sintetizador e do compositor do dashboard
├── pyproject.toml                       # Metadados de empacotamento PEP 517/621
├── requirements.txt                     # Dependências fixadas do ecossistema Python
└── LICENSE                              # Licença de código aberto MIT
```

---

## 🚀 Instalação e Configuração

Requer Python **3.10** ou superior. Recomenda-se criar um ambiente virtual isolado:

```bash
# 1. Clonar o repositório
git clone https://github.com/rafael-b-souza/soccer-ball-detector.git
cd soccer-ball-detector

# 2. Criar e ativar o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar as dependências do projeto
pip install -r requirements.txt

# 4. Instalar o pacote em modo editável
pip install -e .
```

---

## 💻 Como Usar

### 1. Gerar Assets Visuais e Demonstração

Execute o script de demonstração visual para renderizar a partida de futebol sintética com física balística, aplicando o modelo YOLO11 e o rastreador de ROI dinâmica:

```bash
python scripts/generate_demo_visuals.py --frames 45 --fps 15 --output-dir docs/assets
```

Os arquivos `tracking_demo.gif`, `pipeline_comparison.png` e `trajectory_analysis.png` serão atualizados em `docs/assets/`.

---

### 2. Inferência em Python (Quadro a Quadro)

```python
import cv2
from soccer_ball_detector import SoccerBallTracker

# Inicializa o rastreador carregando a configuração ótima do AG
tracker = SoccerBallTracker(
    model_path="yolo11n.pt",
    config_path="configs/best_preproc_params.json"
)

frame = cv2.imread("campo.jpg")

# Processa o quadro (busca localizada na ROI com fallback no frame total)
bbox = tracker.process_frame(frame)

if bbox:
    cx, cy, w, h = bbox
    print(f"Bola localizada em: cx={cx:.1f}, cy={cy:.1f}, w={w:.1f}, h={h:.1f}")
    print(f"Modo de detecção ativo: {tracker.last_mode.upper()}")

# Renderiza anotação visual com bounding box, ROI dinâmica e histórico de trajetória
annotated_frame = tracker.annotate_frame(frame, draw_trajectory=True, show_roi=True)
cv2.imwrite("saida_anotada.jpg", annotated_frame)
```

---

### 3. Anotação e Processamento em Lote de Vídeo

```python
from soccer_ball_detector import annotate_video

metrics = annotate_video(
    input_path="partida.mp4",
    output_path="partida_rastreada.mp4",
    model_path="yolo11n.pt",
    config_path="configs/best_preproc_params.json"
)

print(f"Total de quadros processados: {metrics['frames']}")
print(f"Quadros com bola confirmada: {metrics['detections']}")
print(f"Taxa média efetiva: {metrics['effective_fps']:.2f} FPS")
```

---

### 4. Execução do Algoritmo Genético (DEAP)

```python
from soccer_ball_detector import GeneticOptimizer, compute_fitness

# Configura o otimizador com parâmetros acadêmicos
optimizer = GeneticOptimizer(
    population_size=10,
    generations=10,
    cxpb=0.5,
    mutpb=0.2,
    seed=42
)

# Define sua função de avaliação conectando mAP50 e tempo de inferência
def evaluate(individual):
    # Exemplo: avaliar na base de validação com os parâmetros do cromossomo
    map50 = 0.48
    fps = 62.0
    return (compute_fitness(map50, fps),)

best_individual, logbook, evaluations = optimizer.run(evaluate, verbose=True)
print("Melhor indivíduo descoberto:", best_individual)
```

---

## 🧪 Suíte de Testes Automatizados

A suíte cobre 100% dos módulos do pipeline com 20 testes unitários cobrindo PyTorch, Ultralytics, DEAP, transformações de imagem e computação de ROI:

```bash
pytest -v
```

Saída esperada:
```text
tests/test_detector.py::test_tracker_tensor_preparation PASSED
tests/test_detector.py::test_tracker_roi_and_fallback_flow PASSED
tests/test_detector.py::test_tracker_successful_roi PASSED
tests/test_detector.py::test_tracker_annotation_rendering PASSED
tests/test_optimizer.py::test_fitness_formula_exactness PASSED
tests/test_optimizer.py::test_individual_to_dict_schema PASSED
tests/test_optimizer.py::test_mutation_operator_bounds PASSED
tests/test_optimizer.py::test_safe_crossover_repair PASSED
tests/test_optimizer.py::test_genetic_optimizer_run_with_deap PASSED
tests/test_preproc.py::test_preprocessing_transforms PASSED
tests/test_preproc.py::test_gamma_edge_cases PASSED
tests/test_preproc.py::test_pipeline_all_enabled PASSED
tests/test_preproc.py::test_empty_image_handling PASSED
tests/test_roi.py::test_roi_calculation PASSED
tests/test_roi.py::test_roi_boundary_clamping PASSED
tests/test_roi.py::test_extract_ball_box_empty PASSED
tests/test_roi.py::test_extract_ball_box_with_mock PASSED
tests/test_visuals.py::test_synthetic_match_simulator_determinism PASSED
tests/test_visuals.py::test_visual_demo_composer_output_shape PASSED
tests/test_visuals.py::test_visual_demo_comparison_image PASSED

============================== 20 passed in 2.01s ==============================
```

---

## 📖 Citação Acadêmica

Caso utilize este pipeline ou os hiperparâmetros otimizados em sua pesquisa ou projeto, cite o trabalho correspondente:

```bibtex
@article{souza2025deteccao,
  title   = {Detec{\c{c}}{\~a}o de Bola em Partidas de Futebol: Um Pipeline com YOLO e Otimiza{\c{c}}{\~a}o via Algoritmo Gen{\'e}tico},
  author  = {Souza, Rafael Bernary de and R{\^e}go, Gabriel de Oliveira},
  year    = {2025},
  month   = {fevereiro},
  url     = {https://github.com/rafael-b-souza/soccer-ball-detector}
}
```

---

## 📄 Licença

Este projeto é distribuído sob os termos da licença **MIT**. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
