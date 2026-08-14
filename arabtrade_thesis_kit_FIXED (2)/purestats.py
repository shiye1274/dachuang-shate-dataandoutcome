#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""purestats.py — 纯 Python 标准库实现的统计检验（无需 scipy/statsmodels/pandas）。

t 检验、单/双因素 ANOVA、Wilcoxon 符号秩检验的统计量与 scipy 同公式；
p 值由不完全 beta 函数 / 误差函数精确计算（Numerical Recipes 算法）。
"""
import math
import itertools
from collections import Counter


def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3.0e-14, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a, b, x):
    """正则化不完全 beta 函数 I_x(a,b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t, df):
    """双尾 t 检验 p 值。"""
    if df <= 0:
        return None
    x = df / (df + t * t)
    return betai(df / 2.0, 0.5, x)


def f_sf(F, df1, df2):
    """F 分布上尾 p 值。"""
    if F <= 0:
        return 1.0
    x = df2 / (df2 + df1 * F)
    return betai(df2 / 2.0, df1 / 2.0, x)


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def ttest_rel(a, b):
    """配对 t 检验，返回 (t, p)。"""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return None, None
    m = sum(d) / n
    var = sum((v - m) ** 2 for v in d) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return None, None
    t = m / (sd / math.sqrt(n))
    return t, t_sf_two_sided(t, n - 1)


def ttest_ind(a, b):
    """Welch 两样本 t 检验，返回 (t, p)。"""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None, None
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return None, None
    t = (ma - mb) / se
    denom = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    if denom == 0:
        return None, None
    df = (va / na + vb / nb) ** 2 / denom
    return t, t_sf_two_sided(t, df)


def f_oneway(*groups):
    """单因素 ANOVA，返回 (F, p)。"""
    groups = [list(g) for g in groups if len(g) > 0]
    k = len(groups)
    if k < 2:
        return None, None
    N = sum(len(g) for g in groups)
    grand = sum(sum(g) for g in groups) / N
    ssb = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups)
    ssw = sum(sum((v - sum(g) / len(g)) ** 2 for v in g) for g in groups)
    df1, df2 = k - 1, N - k
    if df2 <= 0 or ssw == 0:
        return None, None
    F = (ssb / df1) / (ssw / df2)
    return F, f_sf(F, df1, df2)


def wilcoxon(a, b):
    """Wilcoxon 符号秩检验（正态近似 + 平键校正），返回 (W, p, z)。"""
    diffs = [x - y for x, y in zip(a, b)]
    diffs = [d for d in diffs if d != 0]
    n = len(diffs)
    if n < 1:
        return None, None, None
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    wpos = sum(ranks[i] for i in range(n) if diffs[i] > 0)
    wneg = sum(ranks[i] for i in range(n) if diffs[i] < 0)
    W = min(wpos, wneg)
    mu = n * (n + 1) / 4.0
    ties = sum(t ** 3 - t for t in Counter(abs(d) for d in diffs).values())
    var = n * (n + 1) * (2 * n + 1) / 24.0 - ties / 48.0
    if var <= 0:
        return W, None, None
    z = (W - mu) / math.sqrt(var)
    return W, 2.0 * norm_sf(abs(z)), z


def tukey_bonf(gains):
    """纯 Python 事后两两比较（Welch t + Bonferroni 校正）代替 Tukey HSD。"""
    keys = [k for k in gains if len(gains[k]) >= 2]
    if len(keys) < 2:
        return None
    pairs = list(itertools.combinations(keys, 2))
    m = len(pairs)
    rows = []
    for g1, g2 in pairs:
        a, b = gains[g1], gains[g2]
        meandiff = round(sum(b) / len(b) - sum(a) / len(a), 4)
        t, p = ttest_ind(a, b)
        if p is None:
            rows.append({"group1": g1, "group2": g2, "meandiff": meandiff,
                         "p_adj": None, "reject": None})
        else:
            padj = min(1.0, p * m)
            rows.append({"group1": g1, "group2": g2, "meandiff": meandiff,
                         "p_raw": p, "p_adj": padj, "reject": padj < 0.05})
    return {"method": "pure_python_bonferroni_t", "comparisons": rows}


def two_way_anova(records):
    """双因素 ANOVA（非加权均值法 / Winer，适用亍不等格），纯 Python。
    因子 A = 问题类型(qtype)，B = 方法(method)。"""
    rows = []
    for r in records:
        for m, col in [("general_llm", "gen_score"), ("plain_rag", "plain_score"), ("rule_rag", "rule_score")]:
            rows.append((r["type"], m, r[col]))
    A = sorted(set(x[0] for x in rows))
    B = sorted(set(x[1] for x in rows))
    I, J = len(A), len(B)
    if I < 2 or J < 2:
        return {"note": "题型或方法类别不足 2 类，无法做双因素 ANOVA", "method": "pure_python"}
    cells = {}
    for a, b, y in rows:
        cells.setdefault((a, b), []).append(y)
    for a in A:
        for b in B:
            if not cells.get((a, b)):
                return {"note": "存在空单元格，无法做双因素 ANOVA", "method": "pure_python"}
    cellmean = {k: sum(v) / len(v) for k, v in cells.items()}
    n_ij = [len(cells[(a, b)]) for a in A for b in B]
    nh = len(n_ij) / sum(1.0 / n for n in n_ij)
    grand = sum(cellmean.values()) / (I * J)
    amean = {a: sum(cellmean[(a, b)] for b in B) / J for a in A}
    bmean = {b: sum(cellmean[(a, b)] for a in A) / I for b in B}
    ssa = nh * J * sum((amean[a] - grand) ** 2 for a in A)
    ssb = nh * I * sum((bmean[b] - grand) ** 2 for b in B)
    ssab = nh * sum((cellmean[(a, b)] - amean[a] - bmean[b] + grand) ** 2 for a in A for b in B)
    ssw = sum((y - cellmean[(a, b)]) ** 2 for a, b, y in rows)
    N = len(rows)
    dfa, dfb, dfab, dfw = I - 1, J - 1, (I - 1) * (J - 1), N - I * J
    if dfw <= 0 or ssw <= 0:
        return {"note": "残差自由度不足，无法计算 F", "method": "pure_python"}
    msw = ssw / dfw

    def blk(ss, df):
        F = (ss / df) / msw
        return {"F": round(F, 4), "p": f_sf(F, df, dfw)}

    return {
        "C(qtype)": blk(ssa, dfa),
        "C(method)": blk(ssb, dfb),
        "C(qtype):C(method)": blk(ssab, dfab),
        "Residual": {"F": None, "p": None},
        "method": "pure_python_unweighted_means",
    }
