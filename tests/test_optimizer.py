"""Testes unitários para o módulo de otimização evolutiva via Algoritmo Genético (DEAP)."""

import pytest
from soccer_ball_detector.optimizer import (
    PREPROC_OPTIONS,
    TILE_CHOICES,
    GeneticOptimizer,
    compute_fitness,
    individual_to_dict,
    mutate_individual,
    safe_crossover,
)


def test_fitness_formula_exactness():
    """Valida a formulação matemática da função de fitness: 0.90 * mAP50 + 0.10 * (FPS / 30.0)."""
    # Exemplo: mAP50 = 0.50, FPS = 30.0 -> 0.90 * 0.50 + 0.10 * 1.0 = 0.45 + 0.10 = 0.55
    score = compute_fitness(map50=0.50, fps=30.0)
    assert pytest.approx(score, rel=1e-5) == 0.55

    # Exemplo dos parâmetros ótimos do artigo: mAP50 = 0.469659, FPS = 61.809085 -> Score ≈ 0.6287
    score_paper = compute_fitness(map50=0.4696594475947594, fps=61.80908558319589)
    assert pytest.approx(score_paper, rel=1e-4) == 0.6287237


def test_individual_to_dict_schema():
    """Valida a serialização do cromossomo do DEAP para o esquema JSON de configuração."""
    sample_ind = [3, 1, 106, 156, 2.84, (16, 16), 1.21, 4.27]
    d = individual_to_dict(sample_ind, score=0.6287, map50=0.47, fps=61.8)

    assert d["preproc"] == "gamma"
    assert d["remove_bg"] is True
    assert d["lower_blue"] == 106
    assert d["upper_blue"] == 156
    assert d["clahe_clip"] == 2.84
    assert d["tile_size"] == [16, 16]
    assert d["gamma_value"] == 1.21
    assert d["roi_scale"] == 4.27
    assert d["score"] == 0.6287


def test_mutation_operator_bounds():
    """Garante que o operador de mutação mantém todos os genes dentro de seus limites permitidos."""
    ind = [0, 0, 100, 145, 2.0, (8, 8), 1.0, 3.0]

    # Força mutação com indpb=1.0 em múltiplas iterações
    for _ in range(20):
        mut_ind = mutate_individual(list(ind), indpb=1.0)[0]

        assert 0 <= mut_ind[0] < len(PREPROC_OPTIONS)
        assert mut_ind[1] in (0, 1)
        assert 90 <= mut_ind[2] <= 110
        assert 130 <= mut_ind[3] <= 160
        assert 1.0 <= mut_ind[4] <= 4.0
        assert isinstance(mut_ind[5], tuple) and len(mut_ind[5]) == 2
        assert mut_ind[5][0] in TILE_CHOICES and mut_ind[5][1] in TILE_CHOICES
        assert 0.8 <= mut_ind[6] <= 1.5
        assert 2.0 <= mut_ind[7] <= 5.0


def test_safe_crossover_repair():
    """Valida que o operador de crossover corrige tuplas corrompidas de tile_size."""
    ind1 = [1, 1, 95, 140, 2.5, "invalido", 1.1, 3.5]
    ind2 = [2, 0, 105, 150, 3.0, (16, 16), 1.3, 4.0]

    res1, res2 = safe_crossover(ind1, ind2)
    assert isinstance(res1[5], tuple) and len(res1[5]) == 2
    assert isinstance(res2[5], tuple) and len(res2[5]) == 2
    assert res1[5][0] in TILE_CHOICES and res1[5][1] in TILE_CHOICES


def test_genetic_optimizer_run_with_deap():
    """Executa um ciclo evolutivo completo de teste usando DEAP garantindo convergência e estatísticas."""
    optimizer = GeneticOptimizer(
        population_size=6,
        generations=2,
        cxpb=0.6,
        mutpb=0.2,
        tourn_size=2,
        seed=42,
    )

    # Avaliador simulado rápido baseado na proximidade dos parâmetros ótimos do artigo
    def mock_evaluate(individual):
        preproc_idx, remove_bg_flag, _, _, _, _, gamma_val, roi_scale = individual
        score = 0.40
        if preproc_idx == 3:  # gamma
            score += 0.10
        if remove_bg_flag == 1:
            score += 0.05
        score += max(0.0, 0.05 - abs(gamma_val - 1.21) * 0.1)
        score += max(0.0, 0.05 - abs(roi_scale - 4.27) * 0.05)
        return (score,)

    best_ind, logbook, _ = optimizer.run(mock_evaluate, verbose=False)

    assert best_ind is not None
    assert len(best_ind) == 8
    assert hasattr(best_ind, "fitness")
    assert best_ind.fitness.valid
    assert best_ind.fitness.values[0] > 0.40

    # Verifica que o logbook do DEAP gerou estatísticas para cada geração
    assert len(logbook) == 3  # Gen 0, 1, 2
    assert "max" in logbook[0]
    assert "avg" in logbook[0]
