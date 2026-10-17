"""MTMM 分析的测试。

关键是**用已知的构念/方法结构**验证它能把两者分开：
- 纯构念数据 → 判据三应通过
- 纯方法数据 → 判据三必须不通过（否则这个方法没用）
"""

from __future__ import annotations

import numpy as np
import pytest

from finpit.eiv import mtmm_analyze


def simulate(
    *,
    n_texts: int = 800,
    n_constructs: int = 3,
    n_methods: int = 4,
    trait_weight: float = 1.0,
    method_weight: float = 0.0,
    noise: float = 0.6,
    seed: int = 0,
) -> np.ndarray:
    """s[t,c,k] = trait_weight·T[t,c] + method_weight·M[t,k] + noise。"""
    rng = np.random.default_rng(seed)
    trait = rng.standard_normal((n_texts, n_constructs, 1))
    method = rng.standard_normal((n_texts, 1, n_methods))
    noise_arr = rng.standard_normal((n_texts, n_constructs, n_methods)) * noise
    return trait_weight * trait + method_weight * method + noise_arr


# ------------------------------------------------------------------ 形状校验
def test_requires_3d():
    with pytest.raises(ValueError):
        mtmm_analyze(np.zeros((10, 5)))


def test_requires_two_constructs():
    with pytest.raises(ValueError):
        mtmm_analyze(np.zeros((50, 1, 3)))


def test_requires_two_methods():
    with pytest.raises(ValueError):
        mtmm_analyze(np.zeros((50, 3, 1)))


def test_requires_enough_samples():
    with pytest.raises(ValueError):
        mtmm_analyze(np.zeros((5, 3, 3)))


def test_name_length_validated():
    with pytest.raises(ValueError):
        mtmm_analyze(
            simulate(n_constructs=3, n_methods=2),
            construct_names=["a", "b"],
        )


# ------------------------------------------------------ 纯构念：判据应通过
def test_pure_trait_structure_passes_all_criteria():
    """数据只含构念成分时，三个判据都应通过。"""
    scores = simulate(trait_weight=1.0, method_weight=0.0, noise=0.5, seed=1)
    r = mtmm_analyze(scores, construct_names=["情绪", "不确定性", "前瞻性"],
                     method_names=["m1", "m2", "m3", "m4"])

    # 同构念异方法高度一致
    assert r.convergent > 0.5
    # 异构念同方法相关低（构念之间独立生成）
    assert r.method_inflated < r.convergent
    assert r.criterion_convergent
    assert r.criterion_trait_over_method
    assert "构念方差占优" in r.verdict
    assert r.trait_share > 0.5


# ------------------------------------------------------ 纯方法：判据必须失败
def test_pure_method_structure_fails_key_criterion():
    """数据只含方法成分时，判据三必须不通过 —— 这是该模块的存在理由。"""
    scores = simulate(n_constructs=3, n_methods=3, trait_weight=0.0,
                      method_weight=1.0, noise=0.5, seed=2)
    r = mtmm_analyze(scores, construct_names=["a", "b", "c"],
                     method_names=["m1", "m2", "m3"])

    # 同一模型测不同构念高度一致（晕轮效应）
    assert r.method_inflated > 0.5
    # 同构念异方法反而不一致
    assert r.method_inflated > r.convergent
    assert not r.criterion_trait_over_method
    assert "方法方差占优" in r.verdict


def test_mixed_structure_scales_with_weights():
    """方法权重越高，构念占比应单调下降。"""
    shares = []
    for w in (0.0, 0.5, 1.0, 2.0):
        scores = simulate(trait_weight=1.0, method_weight=w, noise=0.5, seed=3)
        shares.append(mtmm_analyze(scores).trait_share)
    assert all(shares[i] > shares[i + 1] for i in range(len(shares) - 1))


# ------------------------------------------------------------------ 分解量
def test_method_excess_isolates_method_component():
    """纯方法数据的 method_excess 应显著为正。"""
    scores = simulate(trait_weight=0.0, method_weight=1.0, noise=0.3, seed=4)
    r = mtmm_analyze(scores)
    assert r.method_excess > 0.3


def test_trait_excess_isolates_trait_component():
    scores = simulate(trait_weight=1.0, method_weight=0.0, noise=0.3, seed=5)
    r = mtmm_analyze(scores)
    assert r.trait_excess > 0.3


def test_counts_of_pair_types():
    """配对数量必须精确等于理论值，不能漏格也不能重复计数。"""
    n_c, n_k = 3, 4
    r = mtmm_analyze(simulate(n_constructs=n_c, n_methods=n_k, seed=6))
    pairs_c = n_c * (n_c - 1) // 2
    pairs_k = n_k * (n_k - 1) // 2

    assert r.mthm_values.size == n_c * pairs_k
    assert r.htmm_values.size == pairs_c * n_k
    assert r.hthm_values.size == pairs_c * n_k * (n_k - 1)


def test_serialisation():
    r = mtmm_analyze(simulate(seed=7), construct_names=["x", "y", "z"])
    payload = r.as_dict()
    assert payload["n_constructs"] == 3
    assert set(payload["criteria"]) == {"convergent", "discriminant", "trait_over_method"}
    assert "MTMM" in r.summary()


def test_summary_lists_all_three_criteria():
    r = mtmm_analyze(simulate(seed=8))
    text = r.summary()
    for marker in ("① 收敛效度存在", "② 构念可区分于噪声", "③ 构念 > 方法"):
        assert marker in text


# ------------------------------------------------------- 逐配对诊断（关键）
def simulate_confounded_pair(
    *,
    n_texts: int = 900,
    n_constructs: int = 3,
    n_methods: int = 4,
    confound: float = 1.5,
    noise: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """让**某一对**构念带上方法特有的共同成分。

    这模拟真实场景：同一模型在语义相邻的两个维度上给出雷同打分，
    使这对构念"看起来相关"，但其实只因为它们由同一个模型测。
    """
    rng = np.random.default_rng(seed)
    trait = rng.standard_normal((n_texts, n_constructs, 1))
    method = rng.standard_normal((n_texts, 1, n_methods))
    halo = rng.standard_normal((n_texts, n_methods))  # 方法特有的晕轮成分
    cube = trait + 0.5 * method + noise * rng.standard_normal(
        (n_texts, n_constructs, n_methods)
    )
    # 只给构念 1 与 2 注入同一个方法的晕轮
    cube[:, 1, :] += confound * halo
    cube[:, 2, :] += confound * halo
    return cube


def test_pair_breakdown_detects_confounded_pair():
    """整体均值可能看不出问题，逐配对必须抓到那一对。"""
    cube = simulate_confounded_pair(seed=11)
    r = mtmm_analyze(cube, construct_names=["独立", "受害A", "受害B"])

    confounded = {f"{d.c1}↔{d.c2}" for d in r.pairs if d.confounded}
    assert "受害A↔受害B" in confounded, f"未捕获混杂配对，实际: {confounded}"
    assert "独立↔受害A" not in confounded
    assert "独立↔受害B" not in confounded

    text = r.summary()
    assert "存在方法假象" in text
    assert "主要来自**同一模型**" in r.pair_breakdown().replace("**", "**")


def test_pair_breakdown_clean_data_flags_nothing():
    cube = simulate(trait_weight=1.0, method_weight=0.0, noise=0.5, seed=12)
    r = mtmm_analyze(cube, construct_names=["a", "b", "c"])
    assert all(not d.confounded for d in r.pairs)
    assert "无配对存在明显方法假象" in r.pair_breakdown()


def test_pair_method_excess_sign():
    """同方法相关应高于异方法（混杂数据）。"""
    cube = simulate_confounded_pair(seed=13)
    r = mtmm_analyze(cube, construct_names=["x", "y", "z"])
    victim = next(d for d in r.pairs if {d.c1, d.c2} == {"y", "z"})
    clean = next(d for d in r.pairs if "x" in {d.c1, d.c2})
    assert victim.method_excess > clean.method_excess + 0.2


def test_pairs_serialised():
    r = mtmm_analyze(simulate_confounded_pair(seed=14), construct_names=["a", "b", "c"])
    payload = r.as_dict()
    assert len(payload["pairs"]) == 3
    assert isinstance(payload["confounded_pairs"], list)
    for p in payload["pairs"]:
        assert set(p) == {
            "c1",
            "c2",
            "htmm_same_method",
            "hthm_diff_method",
            "method_excess",
            "confounded",
        }


def test_pair_count_is_choose_two():
    for n_c in (2, 3, 4, 5):
        r = mtmm_analyze(simulate(n_constructs=n_c, n_methods=3, seed=15))
        assert len(r.pairs) == n_c * (n_c - 1) // 2
