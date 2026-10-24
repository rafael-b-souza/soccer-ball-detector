"""Módulo de otimização evolutiva de hiperparâmetros de pré-processamento via Algoritmo Genético (DEAP).

Motivação Acadêmica e de Engenharia:
O pipeline de detecção e rastreamento da bola de futebol contém um espaço de busca combinatorial
multidimensional formado pelo método fotométrico (none, hist_eq, clahe, gamma), ativação de segmentação
de fundo, limiares no espaço HSV, parâmetros do CLAHE (clip limit e tamanho do grid), expoente gamma
e fator multiplicador da ROI dinâmica (roi_scale).
A biblioteca DEAP (Distributed Evolutionary Algorithms in Python) é empregada para descobrir a
combinação de parâmetros de Pareto que maximiza a precisão de detecção (mAP50) ao mesmo tempo em que
assegura alta taxa de quadros (FPS) para viabilizar inferência em tempo real.
"""

import json
import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from deap import algorithms, base, creator, tools

logger = logging.getLogger(__name__)

PREPROC_OPTIONS = ["none", "hist_eq", "clahe", "gamma"]
TILE_CHOICES = [4, 8, 16]


def compute_fitness(
    map50: float,
    fps: float,
    base_fps: float = 30.0,
    map_weight: float = 0.90,
    fps_weight: float = 0.10,
) -> float:
    """Calcula a função de aptidão (fitness) do Algoritmo Genético.

    Regra Acadêmica (Artigo, Seção 3.3):
    A função de fitness combina 90% de mAP50 e 10% de FPS normalizado por 30 FPS:
        Score = 0.90 * mAP50 + 0.10 * (FPS / 30.0)

    Args:
        map50: Mean Average Precision na interseção sobre união (IoU) de 0.5 [0.0, 1.0].
        fps: Taxa de quadros por segundo medida durante a inferência na validação.
        base_fps: Taxa base de FPS para normalização (padrão: 30.0).
        map_weight: Ponderação da precisão mAP50 (padrão: 0.90).
        fps_weight: Ponderação da velocidade FPS (padrão: 0.10).

    Returns:
        Valor escalar de fitness correspondente à qualidade do indivíduo.
    """
    normalized_fps = max(0.0, fps) / base_fps
    return (map_weight * map50) + (fps_weight * normalized_fps)


def safe_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    """Executa crossover de dois pontos garantindo a integridade estrutural dos genes.

    Regra de Engenharia:
    Após a troca de segmentos cromossômicos por cxTwoPoint, o gene tile_size (índice 5)
    pode ocasionalmente sofrer inconsistência de tipo. Este operador valida e repara
    o formato de tupla (int, int) com escolhas válidas em [4, 8, 16].
    """
    tools.cxTwoPoint(ind1, ind2)
    for ind in [ind1, ind2]:
        if not isinstance(ind[5], tuple) or len(ind[5]) != 2:
            ind[5] = (random.choice(TILE_CHOICES), random.choice(TILE_CHOICES))
    return ind1, ind2


def mutate_individual(individual: list, indpb: float = 0.2) -> Tuple[list]:
    """Aplica mutação estocástica aos genes do indivíduo respeitando os limites do domínio de busca.

    Regra Acadêmica:
    Cada gene possui probabilidade independente indpb de sofrer mutação dentro de sua faixa útil:
      - preproc_idx: {0, 1, 2, 3}
      - remove_bg: {0, 1}
      - lower_blue: [90, 110]
      - upper_blue: [130, 160]
      - clahe_clip: [1.0, 4.0]
      - tile_size: escolhas em [4, 8, 16] x [4, 8, 16]
      - gamma_value: [0.8, 1.5]
      - roi_scale: [2.0, 5.0]
    """
    if random.random() < indpb:
        individual[0] = random.randint(0, len(PREPROC_OPTIONS) - 1)
    if random.random() < indpb:
        individual[1] = random.randint(0, 1)
    if random.random() < indpb:
        individual[2] = random.randint(90, 110)
    if random.random() < indpb:
        individual[3] = random.randint(130, 160)
    if random.random() < indpb:
        individual[4] = random.uniform(1.0, 4.0)
    if random.random() < indpb:
        individual[5] = (random.choice(TILE_CHOICES), random.choice(TILE_CHOICES))
    if random.random() < indpb:
        individual[6] = random.uniform(0.8, 1.5)
    if random.random() < indpb:
        individual[7] = random.uniform(2.0, 5.0)
    return (individual,)


def individual_to_dict(
    individual: list,
    score: Optional[float] = None,
    map50: Optional[float] = None,
    fps: Optional[float] = None,
    training_time: Optional[float] = None,
) -> Dict[str, Any]:
    """Converte o vetor de cromossomos do DEAP no esquema de dicionário estruturado do JSON de configuração.

    Args:
        individual: Lista com os 8 genes do cromossomo.
        score: Pontuação de fitness atribuída.
        map50: Métrica mAP50 obtida.
        fps: Taxa de quadros registrada.
        training_time: Tempo de treinamento/avaliação em segundos.

    Returns:
        Dicionário formatado conforme configs/best_preproc_params.json.
    """
    preproc_idx, remove_bg_flag, lower_blue, upper_blue, clahe_clip, tile_size, gamma_val, roi_scale = individual
    return {
        "preproc": PREPROC_OPTIONS[int(preproc_idx)],
        "remove_bg": bool(remove_bg_flag),
        "lower_blue": int(lower_blue),
        "upper_blue": int(upper_blue),
        "clahe_clip": float(clahe_clip),
        "tile_size": list(tile_size),
        "gamma_value": float(gamma_val),
        "roi_scale": float(roi_scale),
        "training_time": float(training_time) if training_time is not None else 0.0,
        "mAP50": float(map50) if map50 is not None else 0.0,
        "fps": float(fps) if fps is not None else 0.0,
        "score": float(score) if score is not None else (
            individual.fitness.values[0] if hasattr(individual, "fitness") and individual.fitness.valid else 0.0
        ),
    }


class GeneticOptimizer:
    """Gerenciador do Algoritmo Genético usando DEAP para o pipeline de detecção e rastreamento."""

    def __init__(
        self,
        population_size: int = 10,
        generations: int = 10,
        cxpb: float = 0.5,
        mutpb: float = 0.2,
        tourn_size: int = 3,
        seed: Optional[int] = 42,
    ):
        """Inicializa o otimizador genético.

        Args:
            population_size: Número de indivíduos por geração (padrão: 10, conforme artigo).
            generations: Número de gerações evolutivas (padrão: 10, conforme artigo).
            cxpb: Probabilidade de crossover entre indivíduos (padrão: 0.5).
            mutpb: Probabilidade de mutação por indivíduo (padrão: 0.2).
            tourn_size: Tamanho do torneio para seleção (padrão: 3).
            seed: Semente pseudo-aleatória para reprodutibilidade estrita.
        """
        self.pop_size = population_size
        self.ngen = generations
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.tourn_size = tourn_size

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._setup_deap()
        self.toolbox = self._create_toolbox()
        self.evaluations_log: List[Dict[str, Any]] = []

    def _setup_deap(self):
        """Configura classes do DEAP com checagem de existência para evitar colisões em múltiplas instâncias."""
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

    def _create_toolbox(self) -> base.Toolbox:
        """Registra os operadores genéticos e geradores de genes na toolbox do DEAP."""
        tb = base.Toolbox()
        tb.register("attr_preproc", lambda: random.randint(0, len(PREPROC_OPTIONS) - 1))
        tb.register("attr_remove_bg", lambda: random.randint(0, 1))
        tb.register("attr_lower_blue", lambda: random.randint(90, 110))
        tb.register("attr_upper_blue", lambda: random.randint(130, 160))
        tb.register("attr_clahe_clip", lambda: random.uniform(1.0, 4.0))
        tb.register("attr_tile_size", lambda: (random.choice(TILE_CHOICES), random.choice(TILE_CHOICES)))
        tb.register("attr_gamma_value", lambda: random.uniform(0.8, 1.5))
        tb.register("attr_roi_scale", lambda: random.uniform(2.0, 5.0))

        tb.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (
                tb.attr_preproc,
                tb.attr_remove_bg,
                tb.attr_lower_blue,
                tb.attr_upper_blue,
                tb.attr_clahe_clip,
                tb.attr_tile_size,
                tb.attr_gamma_value,
                tb.attr_roi_scale,
            ),
            n=1,
        )
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("mate", safe_crossover)
        tb.register("mutate", mutate_individual, indpb=self.mutpb)
        tb.register("select", tools.selTournament, tournsize=self.tourn_size)
        return tb

    def run(
        self,
        eval_fn: Callable[[list], Tuple[float]],
        verbose: bool = False,
    ) -> Tuple[list, tools.Logbook, List[Dict[str, Any]]]:
        """Executa a evolução por gerações com a função de avaliação informada.

        Args:
            eval_fn: Função com assinatura eval_fn(individual) -> (score,).
            verbose: Se True, emite estatísticas parciais por geração via logging.

        Returns:
            Tupla contendo:
              - best_ind: Melhor indivíduo encontrado (creator.Individual).
              - logbook: Histórico estatístico da evolução (DEAP Logbook).
              - evaluations_log: Lista com detalhes de cada indivíduo avaliado.
        """
        self.toolbox.register("evaluate", eval_fn)
        pop = self.toolbox.population(n=self.pop_size)

        stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        hof = tools.HallOfFame(1)

        pop, logbook = algorithms.eaSimple(
            pop,
            self.toolbox,
            cxpb=self.cxpb,
            mutpb=self.mutpb,
            ngen=self.ngen,
            stats=stats,
            halloffame=hof,
            verbose=verbose,
        )

        best_ind = hof[0]
        return best_ind, logbook, self.evaluations_log
