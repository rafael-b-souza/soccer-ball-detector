# Soccer Ball Detector

Pipeline de visao computacional para deteccao e rastreamento de bola em partidas de futebol, combinando a arquitetura YOLO11, tecnicas de pre-processamento de imagem otimizadas por Algoritmo Genetico (DEAP) e rastreamento com regiao de interesse (ROI) dinamica.

O projeto foi desenvolvido no contexto academico por Rafael Bernary de Souza e Gabriel de Oliveira Rêgo. O artigo correspondente com a fundamentacao teorica, metodologia e analise experimental esta disponivel em [`docs/deteccao_de_bola_em_partidas_de_futebol.pdf`](docs/deteccao_de_bola_em_partidas_de_futebol.pdf).

## Contexto e Desafio

A deteccao da bola em videos esportivos enfrenta obstaculos frequentes:
- Pequenas dimensoes relativas do objeto em relacao a resolucao do campo.
- Oclusoes por jogadores e mudancas abruptas de iluminacao e contraste.
- Variacao de tonalidades do gramado e das linhas do campo.
- Perdas de confianca durante movimentos rapidos e desfoque por movimento (motion blur).

Para contornar essas restricoes sem inflacionar o custo computacional, o pipeline adota uma abordagem em duas fases:
1. **Otimizacao Evolutiva de Hiperparametros de Imagem**: Um Algoritmo Genetico (AG) explora espacos de combinacoes de equalizacao (CLAHE, correcao gamma, segmentacao HSV) maximizando uma funcao de aptidao equilibrada entre mAP50 e taxa de quadros (FPS).
2. **Rastreamento com ROI Dinamica**: Apos uma deteccao confirmada no frame completo, os frames subsequentes focam em uma regiao de interesse redimensionada ao redor da ultima posicao conhecida. Em caso de perda, o algoritmo faz fallback automatico para o quadro inteiro.

## Resultados do Algoritmo Genetico

A configuracao otimizada encontrada pelo AG esta registrada em [`configs/best_preproc_params.json`](configs/best_preproc_params.json):

| Parametro | Valor Selecionado |
| :--- | :--- |
| Metodo de Pre-processamento | Correcao Gamma (\`gamma\`) |
| Valor Gamma ($\gamma$) | ~1.21 |
| Segmentacao de Fundo | Ativada (\`remove_bg = True\`) |
| Faixa HSV | Blue bounds: 106 a 156 |
| Escala da ROI (\`roi_scale\`) | ~4.27x |
| CLAHE Clip Limit (candidato) | 2.84 |
| Tile Size | 16x16 |

O historico completo de individuos avaliados ao longo das geracoes encontra-se em [`data/ga_detailed_evaluations.csv`](data/ga_detailed_evaluations.csv).

## Estrutura do Repositorio

```
soccer-ball-detector/
├── configs/
│   └── best_preproc_params.json         # Parametros otimizados pelo AG
├── data/
│   └── ga_detailed_evaluations.csv      # Log de convergencia do GA
├── docs/
│   └── deteccao_de_bola_em_partidas_de_futebol.pdf # Artigo academico
├── notebooks/
│   └── pipeline_deteccao_bola.ipynb     # Notebook Jupyter / Colab completo
├── src/
│   └── soccer_ball_detector/
│       ├── preproc.py                   # Transformacoes CLAHE, gamma, HSV
│       ├── roi.py                       # Calculo e extracao de ROI
│       ├── detector.py                  # Tracker integrado com fallback
│       └── video.py                     # Leitura, anotacao e gravacao de video
├── tests/
│   └── test_preproc.py                  # Testes unitarios de transformacoes
└── pyproject.toml
```

## Instalacao

Recomenda-se Python 3.10 ou superior em ambiente virtual dedicado:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Como Usar

### Execucao em Python

```python
import cv2
from soccer_ball_detector.detector import SoccerBallTracker

# Inicializa o rastreador carregando os parametros otimizados
tracker = SoccerBallTracker(
    model_path="yolov11-small.pt",
    config_path="configs/best_preproc_params.json"
)

frame = cv2.imread("campo.jpg")
bbox = tracker.process_frame(frame)

if bbox:
    cx, cy, w, h = bbox
    print(f"Bola detectada em: cx={cx:.1f}, cy={cy:.1f}, w={w:.1f}, h={h:.1f}")
```

### Processamento de Video

```python
from soccer_ball_detector.video import annotate_video

stats = annotate_video(
    input_path="partida.mp4",
    output_path="saida_anotada.mp4",
    model_path="yolov11-small.pt",
    config_path="configs/best_preproc_params.json"
)
print(stats)
```

### Notebook Interativo

O pipeline de ponta a ponta (download de frames, treinamento da rede, execucao do GA e geracao de graficos) pode ser executado diretamente em [`notebooks/pipeline_deteccao_bola.ipynb`](notebooks/pipeline_deteccao_bola.ipynb) no Jupyter Lab ou Google Colab.

## Testes

Para executar os testes unitarios:

```bash
pytest
```

## Licenca

Este projeto esta sob a licenca MIT. Consulte o arquivo [LICENSE](LICENSE) para mais informacoes.
