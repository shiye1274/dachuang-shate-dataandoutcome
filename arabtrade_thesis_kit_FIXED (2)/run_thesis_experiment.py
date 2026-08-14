#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_thesis_experiment.py — 确定性 / 可审计的论文实验运行器

为什么用它（防 AI 幻觉的 5 条硬保证）：
1. 绝不静默回退：连不上 DeepSeek API 直接报错退出，不会用模板假装结果。
2. 全量 + 分层：默认跑完整评测集；可用 --max-q / --per-scenario 限制；按场景分层。
3. 真实统计检验：H1 配对 t 检验、H2 单因素 ANOVA(+Tukey 事后)、H3 Wilcoxon、
   H5 双因素 ANOVA，t/F/Z/p 全部由 scipy/statsmodels 从数据算出，绝不写死。
4. 全程留痕：每题每条件的原始回答与打分写入 audit CSV，任何数字都能回溯核对。
5. 错误分析按实际结果统计，不使用写死常数。

用法：
    export DEEPSEEK_API_KEY=sk-xxxx
    python run_thesis_experiment.py                  # 跑全量 833 条
    python run_thesis_experiment.py --per-scenario 100
    python run_thesis_experiment.py --max-q 150
    python run_thesis_experiment.py --dry-run        # 不连API，仅自检脚本与统计管线

依赖： 无需任何第三方库！统计用纯 Python 实现，API 调用用标准库 urllib。
       （若已装 scipy/statsmodels 会自动用它们交叉验证，但不装也能算出等价结果。）
"""
import os, sys, csv, json, math, time, argparse, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation_dataset import get_evaluation_dataset, get_dataset_stats
from experiment_runner import LLMClient, KnowledgeBase
from rule_engine import create_rule_engine
from thesis_results import compute_factual_accuracy, compute_four_segment_score
from knowledge_base.compliance import get_compliance_knowledge
from knowledge_base.market import get_market_knowledge
from knowledge_base.platform import get_platform_knowledge
from knowledge_base.marketing import get_marketing_knowledge
from knowledge_base.risk import get_risk_knowledge

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

AUDIT_COLS = ["id", "scenario", "type", "source", "rule_hit", "retrieved_count",
              "gen_score", "plain_score", "rule_score",
              "gen_cover", "plain_cover", "rule_cover",
              "gen_judge", "plain_judge", "rule_judge",
              "gen_interp", "plain_interp", "rule_interp",
              "question", "standard_answer", "gen_answer", "plain_answer", "rule_answer"]


def _coerce_record(r):
    """CSV 断点文件读回后，把数字/布尔字段恢复成统计函数需要的类型。"""
    out = dict(r)
    for k in ["retrieved_count"]:
        try:
            out[k] = int(out.get(k, 0) or 0)
        except Exception:
            out[k] = 0
    for k in ["gen_score", "plain_score", "rule_score", "gen_cover", "plain_cover", "rule_cover",
              "gen_interp", "plain_interp", "rule_interp"]:
        try:
            out[k] = float(out.get(k, 0) or 0)
        except Exception:
            out[k] = 0.0
    for k in ["gen_judge", "plain_judge", "rule_judge"]:
        v = out.get(k, "")
        if v == "" or v is None:
            out[k] = ""
        else:
            try:
                out[k] = float(v)
            except Exception:
                out[k] = ""
    out["rule_hit"] = str(out.get("rule_hit", "")).lower() in ("true", "1", "yes", "y")
    return out


def load_progress_records(progress_path, valid_ids):
    """读取已完成题目，用于断点续跑；只保留当前评测子集内的 id。"""
    done = {}
    if not progress_path or not os.path.exists(progress_path):
        return done
    try:
        with open(progress_path, "r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                rid = r.get("id", "")
                if rid in valid_ids:
                    done[rid] = _coerce_record(r)
    except Exception as e:
        print("[WARN] 读取断点文件失败，将从头开始。原因：%s" % e)
        return {}
    return done


def append_progress_record(progress_path, rec):
    """每完成一题立即落盘；即使后面断网，已完成部分也不会丢。"""
    if not progress_path:
        return
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)
    need_header = (not os.path.exists(progress_path)) or os.path.getsize(progress_path) == 0
    with open(progress_path, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_COLS)
        if need_header:
            w.writeheader()
        w.writerow({k: rec.get(k, "") for k in AUDIT_COLS})
        f.flush()



class StrictLLMClient(LLMClient):
    """连不上就抛错，绝不回退模板 —— 这是防幻觉的关键。"""
    def _fallback(self, messages, reason="No API key"):
        raise RuntimeError(
            "LLM 调用失败，已中止以避免产生虚假结果。原因: " + str(reason) +
            "\n请检查 DEEPSEEK_API_KEY / 网络 / 余额后重试。")

    def generate(self, messages, temperature=0.3, max_tokens=1024):
        # 仅用标准库 urllib 调用，避免依赖 requests。临时网络错误会自动重试。
        if not self.api_key:
            return self._fallback(messages)
        import json as _json
        import urllib.request as _urlreq
        import urllib.error as _urlerr
        body = _json.dumps({
            "model": self.model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens,
        }).encode("utf-8")
        req = _urlreq.Request(
            self.base_url + "/v1/chat/completions", data=body,
            headers={"Authorization": "Bearer " + self.api_key,
                     "Content-Type": "application/json"})
        last_err = None
        waits = [3, 8, 15, 30, 60, 90]
        for attempt, wait_s in enumerate([0] + waits, start=1):
            if wait_s:
                print("\n[网络重试] 第 %d 次重试，等待 %d 秒后继续。" % (attempt - 1, wait_s))
                time.sleep(wait_s)
            try:
                with _urlreq.urlopen(req, timeout=90) as resp:
                    data = _json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                msg = str(e)
                if isinstance(e, _urlerr.HTTPError) and e.code in (400, 401, 403):
                    break
                print("\n[WARN] LLM 调用暂时失败：%s" % msg[:180])
        return self._fallback(messages, str(last_err))


def preflight(llm):
    """开跑前先探一次 API，不通就退出。"""
    if not llm.api_key:
        sys.exit("[ABORT] 未检测到 DEEPSEEK_API_KEY，拒绝以回退模板伪造结果。\n"
                 "        请先 export DEEPSEEK_API_KEY=sk-xxxx（或用 --dry-run 仅自检）。")
    print("[preflight] 正在测试 DeepSeek API 连通性 ...")
    txt = llm.generate([{"role": "user", "content": "回复两个字：正常"}], max_tokens=8)
    print("[preflight] API 返回:", repr(txt[:40]))
    print("[preflight] OK, 开始正式实验。\n")


def select_questions(dataset, per_scenario=None, max_q=None):
    by_sc = {}
    for it in dataset:
        by_sc.setdefault(it["scenario"], []).append(it)
    selected = []
    for sc, items in by_sc.items():
        selected.extend(items if per_scenario is None else items[:per_scenario])
    if max_q is not None:
        selected = selected[:max_q]
    return selected


def kb_size():
    return (len(get_compliance_knowledge()) + len(get_market_knowledge()) +
            len(get_platform_knowledge()) + len(get_marketing_knowledge()) +
            len(get_risk_knowledge()))


def llm_judge(llm, question, standard, answer):
    """用大模型当评委，依据标准答案给 0-100 分并归一到 0-1。仅评分，不生成正文。"""
    txt = llm.generate([
        {"role": "system", "content":
            "你��严格公正的评分员。依据【标准答案】判断【待评回答】的事实正确性与是否切题，"
            "只输出一个 0 到 100 的整数分数，不要输出任何其它文字或符号。"
            "切题且与标准答案一致给高分；答非所问或遗漏关键点给低分；编造与标准答案矛盾的信息给接近 0 分。"},
        {"role": "user", "content":
            "问题：%s\n\n标准答案：%s\n\n待评回答：%s\n\n只输出 0-100 的整数：" % (question, standard, answer)},
    ], temperature=0.0, max_tokens=8)
    import re as _re
    m = _re.search(r"\d{1,3}", txt or "")
    if not m:
        return 0.0
    v = min(100, int(m.group()))
    return max(0.0, v / 100.0)


def run(dataset, llm, judge=False, dry=False, progress_path=None):
    kb = KnowledgeBase()
    rule_engine = create_rule_engine()
    records = []
    n = len(dataset)
    valid_ids = set(item["id"] for item in dataset)
    completed = {} if dry else load_progress_records(progress_path, valid_ids)
    if completed:
        print("[断点续跑] 已找到 %d/%d 条已完成记录，将自动跳过这些题。" % (len(completed), n))
    for idx, item in enumerate(dataset):
        if item["id"] in completed:
            records.append(completed[item["id"]])
            if (len(records)) % 10 == 0 or len(records) == n:
                print("  进度 %d/%d（已跳过完成项）" % (len(records), n))
            continue
        q, std = item["question"], item["standard_answer"]
        print("  [%d/%d] 正在处理 %s（%s）..." % (idx + 1, n, item["id"], item["scenario"]), flush=True)
        # 条件1：通用 LLM
        gen = llm.generate([
            {"role": "system", "content": "你是沙特出海贸易顾问，回答中国中小卖家关于沙特服装市场的问题。给出简要直接的回答。"},
            {"role": "user", "content": q}])
        # 条件2：纯文本 RAG
        retrieved = kb.search(q, top_k=5)
        context = "\n\n".join("[%d] %s: %s" % (i + 1, r.get("title", ""), r.get("content", "")[:200])
                              for i, r in enumerate(retrieved))
        plain = llm.generate([
            {"role": "system", "content": "你是沙特市场顾问。请基于以下参考资料回答沙特服装市场的问题，引用来源编号。"},
            {"role": "user", "content": "问题：%s\n\n参考资料：\n%s" % (q, context)}])
        # 条件3：规则增强 RAG —— 规则要点注入提示，让模型【先正面回答问题】再给合规护栏
        rule_out = rule_engine.get_four_segment_output(q)
        rule_hit = bool(rule_out["has_rule_hit"])
        rf = []
        if rule_hit:
            rf.append("【命中合规规则】" + rule_out["compliance_reminder"])
            if rule_out["action_suggestions"]:
                rf.append("建议措施：" + "；".join(rule_out["action_suggestions"][:3]))
            if rule_out["risk_alerts"]:
                rf.append("风险要点：" + "；".join(rule_out["risk_alerts"][:3]))
            if rule_out["citations"]:
                rf.append("权威依据：" + "；".join(rule_out["citations"][:3]))
        rule_facts = "\n".join(rf) if rf else "（本题未命中硬性合规规则，请正常作答）"
        rule_ans = llm.generate([
            {"role": "system", "content":
                "你是沙特出海市场顾问。请【先用一两句正面、具体地回答用户的问题本身】，"
                "再结合规则要点给出合规提醒、行动建议、风险提示与引用来源。"
                "必须真正回答问题，严禁只堆砌与问题无关的通用合规套话；规则要点与问题相关才写入，不相关就省略。"},
            {"role": "user", "content":
                "问题：%s\n\n参考资料：\n%s\n\n可参考的规则要点：\n%s\n\n"
                "请按此结构作答：\n## 直接回答\n（先正面回答上面的问题）\n## 合规提醒\n## 行动建议\n## 风险提示\n## 引用来源"
                % (q, context, rule_facts)}])
        cover_gen = compute_factual_accuracy(gen, std)
        cover_plain = compute_factual_accuracy(plain, std)
        cover_rule = compute_factual_accuracy(rule_ans, std)
        if judge:
            j_gen = llm_judge(llm, q, std, gen)
            j_plain = llm_judge(llm, q, std, plain)
            j_rule = llm_judge(llm, q, std, rule_ans)
            sc_gen, sc_plain, sc_rule = j_gen, j_plain, j_rule
        else:
            j_gen = j_plain = j_rule = ""
            sc_gen, sc_plain, sc_rule = cover_gen, cover_plain, cover_rule
        rec = {
            "id": item["id"], "question": q, "standard_answer": std,
            "type": item["type"], "scenario": item["scenario"], "source": item.get("source", ""),
            "retrieved_count": len(retrieved), "rule_hit": rule_hit,
            "gen_answer": gen, "plain_answer": plain, "rule_answer": rule_ans,
            "gen_score": sc_gen, "plain_score": sc_plain, "rule_score": sc_rule,
            "gen_cover": cover_gen, "plain_cover": cover_plain, "rule_cover": cover_rule,
            "gen_judge": j_gen, "plain_judge": j_plain, "rule_judge": j_rule,
            "gen_interp": compute_four_segment_score(gen),
            "plain_interp": compute_four_segment_score(plain),
            "rule_interp": compute_four_segment_score(rule_ans),
        }
        records.append(rec)
        append_progress_record(progress_path, rec)
        print("  [%d/%d] 完成 ✓" % (len(records), n), flush=True)
    return records


# ---------------- 统计（全部真实计算） ----------------
def _mean(x):
    return sum(x) / len(x) if x else 0.0


def _std(x):
    m = _mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / len(x)) if x else 0.0


def compute_stats(records):
    try:
        from scipy import stats as st
    except Exception:
        st = None
    import purestats as PS
    out = {}
    out["_scipy_available"] = st is not None
    out["_inferential_method"] = "scipy" if st is not None else "pure_python"
    gen = [r["gen_score"] for r in records]
    plain = [r["plain_score"] for r in records]
    rule = [r["rule_score"] for r in records]
    gi = [r["gen_interp"] for r in records]
    ri = [r["rule_interp"] for r in records]

    # H1 主对照 + 配对 t 检验
    def paired(a, b):
        d = [x - y for x, y in zip(a, b)]
        sd = _std(d)
        base = {"df": len(a) - 1, "t": None, "p": None,
                "mean_diff": round(_mean(d), 4),
                "cohen_d": round(_mean(d) / sd, 4) if sd else None}
        if st is not None:
            t, p = st.ttest_rel(a, b)
            base["method"] = "scipy"
        else:
            t, p = PS.ttest_rel(a, b)
            base["method"] = "pure_python"
        if t is not None:
            base["t"] = round(float(t), 4)
            base["p"] = float(p) if p is not None else None
        return base
    out["H1"] = {
        "n": len(records),
        "general_llm": {"acc_mean": round(_mean(gen), 4), "acc_std": round(_std(gen), 4), "interp_mean": round(_mean(gi), 4)},
        "plain_rag": {"acc_mean": round(_mean(plain), 4), "acc_std": round(_std(plain), 4)},
        "rule_rag": {"acc_mean": round(_mean(rule), 4), "acc_std": round(_std(rule), 4), "interp_mean": round(_mean(ri), 4)},
        "ttest_rule_vs_general": paired(rule, gen),
        "ttest_rule_vs_plain": paired(rule, plain),
    }

    # H2 分场景增益 + 单因素 ANOVA + Tukey
    scenarios = ["compliance", "marketing", "product"]
    gains, by_sc = {}, {}
    for sc in scenarios:
        rs = [r for r in records if r["scenario"] == sc]
        if not rs:
            continue
        g = [r["rule_score"] - r["gen_score"] for r in rs]
        gains[sc] = g
        by_sc[sc] = {"n": len(rs),
                     "general_llm": round(_mean([r["gen_score"] for r in rs]), 4),
                     "plain_rag": round(_mean([r["plain_score"] for r in rs]), 4),
                     "rule_rag": round(_mean([r["rule_score"] for r in rs]), 4),
                     "gain_vs_general": round(_mean(g), 4)}
    anova = None
    if len(gains) >= 2 and all(len(v) >= 2 for v in gains.values()):
        if st is not None:
            F, p = st.f_oneway(*gains.values())
            meth = "scipy"
        else:
            F, p = PS.f_oneway(*gains.values())
            meth = "pure_python"
        if F is not None:
            anova = {"F": round(float(F), 4), "p": float(p), "groups": list(gains.keys()), "method": meth}
    tukey = _tukey(gains)
    out["H2"] = {"by_scenario": by_sc, "anova": anova, "tukey": tukey}

    # H3 可解释性 + Wilcoxon
    def interp_block(vals):
        s = sorted(vals)
        return {"mean": round(_mean(vals), 4), "std": round(_std(vals), 4),
                "median": round(s[len(s) // 2], 4) if s else 0}
    wil = None
    try:
        diffs = [a - b for a, b in zip(ri, gi)]
        nnz = sum(1 for d in diffs if d != 0)
        if nnz == 0:
            wil = {"note": "rule 与 general 的可解释性差异全为 0，无法做 Wilcoxon"}
        elif st is not None:
            W, p = st.wilcoxon(ri, gi, zero_method="wilcox")
            mu = nnz * (nnz + 1) / 4.0
            sig = math.sqrt(nnz * (nnz + 1) * (2 * nnz + 1) / 24.0)
            z = (float(W) - mu) / sig if sig else None
            wil = {"W": round(float(W), 4), "p": float(p), "z": round(z, 4) if z is not None else None, "n_nonzero": nnz, "method": "scipy"}
        else:
            W, p, z = PS.wilcoxon(ri, gi)
            wil = {"W": round(float(W), 4), "p": float(p) if p is not None else None, "z": round(float(z), 4) if z is not None else None, "n_nonzero": nnz, "method": "pure_python"}
    except Exception as e:
        wil = {"error": str(e)}
    out["H3"] = {"general_llm": interp_block(gi), "plain_rag": interp_block([r["plain_interp"] for r in records]),
                 "rule_rag": interp_block(ri), "wilcoxon_rule_vs_general": wil}

    # H5 问题类型 × 方法 双因素 ANOVA
    types = ["fact", "compare", "reasoning"]
    by_t = {}
    for qt in types:
        rs = [r for r in records if r["type"] == qt]
        if not rs:
            continue
        by_t[qt] = {"n": len(rs),
                    "general_llm": round(_mean([r["gen_score"] for r in rs]), 4),
                    "plain_rag": round(_mean([r["plain_score"] for r in rs]), 4),
                    "rule_rag": round(_mean([r["rule_score"] for r in rs]), 4),
                    "rule_gain": round(_mean([r["rule_score"] - r["gen_score"] for r in rs]), 4)}
    out["H5"] = {"by_type": by_t, "two_way_anova": _two_way(records)}

    # 6.6 错误分析（按实际结果分类，非写死）
    out["error_analysis"] = _errors(records)
    out["total_questions"] = len(records)
    out["kb_size"] = kb_size()
    return out


def _tukey(gains):
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        import numpy as np
        vals, labs = [], []
        for k, v in gains.items():
            vals += v
            labs += [k] * len(v)
        if len(set(labs)) < 2:
            return None
        res = pairwise_tukeyhsd(np.array(vals), np.array(labs))
        rows = []
        for line in res.summary().data[1:]:
            rows.append({"group1": line[0], "group2": line[1], "meandiff": float(line[2]),
                         "p_adj": float(line[3]), "reject": bool(line[6])})
        return rows
    except Exception:
        import purestats as PS
        return PS.tukey_bonf(gains)


def _two_way(records):
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
        rows = []
        for r in records:
            for m, col in [("general_llm", "gen_score"), ("plain_rag", "plain_score"), ("rule_rag", "rule_score")]:
                rows.append({"score": r[col], "qtype": r["type"], "method": m})
        df = pd.DataFrame(rows)
        if df["qtype"].nunique() < 2 or df["method"].nunique() < 2:
            return {"note": "题型或方法类别不足 2 类，无法做双因素 ANOVA"}
        model = smf.ols("score ~ C(qtype) * C(method)", data=df).fit()
        aov = anova_lm(model, typ=2)
        res = {}
        for term in aov.index:
            res[str(term)] = {"F": None if math.isnan(aov.loc[term, "F"]) else round(float(aov.loc[term, "F"]), 4),
                              "p": None if math.isnan(aov.loc[term, "PR(>F)"]) else float(aov.loc[term, "PR(>F)"])}
        return res
    except Exception:
        import purestats as PS
        return PS.two_way_anova(records)


def _errors(records):
    cats = {"correct_or_partial": 0, "retrieval_fail": 0, "rule_missing": 0,
            "hallucination": 0, "annotation_dispute": 0}
    hedge = ["资料不足", "无法确认", "不确定", "未知"]
    for r in records:
        ans, sc = r["rule_answer"], r["rule_score"]
        if sc >= 0.5:
            cats["correct_or_partial"] += 1
        elif r["retrieved_count"] == 0:
            cats["retrieval_fail"] += 1
        elif (not r["rule_hit"]) and r["scenario"] == "compliance":
            cats["rule_missing"] += 1
        elif not any(h in ans for h in hedge):
            cats["hallucination"] += 1
        else:
            cats["annotation_dispute"] += 1
    tot = sum(cats.values()) or 1
    return {k: {"count": v, "pct": round(100.0 * v / tot, 1)} for k, v in cats.items()}


def save_audit(records, path):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_COLS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in AUDIT_COLS})


def pfmt(p):
    if p is None:
        return "n/a"
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return "p<0.01"
    if p < 0.05:
        return "p<0.05"
    return "p=%.3f" % p


def chapter(stats, meta):
    L = []
    H1, H2, H3, H5 = stats["H1"], stats["H2"], stats["H3"], stats["H5"]
    L.append("# 第六章 实验结果与分析（自动生成，数字均由脚本计算）")
    L.append("")
    L.append("> 运行时间 %s ｜ 模型 %s ｜ API可用 %s ｜ 评分口径 %s ｜ 评测集 %d 条 ｜ 知识库 %d 条" % (
        meta["time"], meta["model"], meta["api_available"], meta.get("scoring", "关键点覆盖率"), stats["total_questions"], stats["kb_size"]))
    L.append("")
    g, p, r = H1["general_llm"], H1["plain_rag"], H1["rule_rag"]
    tt = H1["ttest_rule_vs_general"]
    L.append("## 6.1 主对照结果（H1）")
    tstr = ("t(%d)=%.2f, %s" % (tt["df"], tt["t"], pfmt(tt["p"]))) if tt.get("t") is not None else "t 检验待 scipy 环境计算"
    L.append("规则增强 RAG 事实准确率 %.1f%%，通用 LLM %.1f%%，配对 t 检验 %s，差值 %.1f 个百��点。" % (
        r["acc_mean"] * 100, g["acc_mean"] * 100, tstr, tt["mean_diff"] * 100))
    L.append("")
    L.append("| 条件 | 样本量 | 事实准确率 | 标准差 |")
    L.append("|------|-----|-----|-----|")
    L.append("| 通用 LLM | %d | %.1f%% | %.3f |" % (H1["n"], g["acc_mean"] * 100, g["acc_std"]))
    L.append("| 纯文本 RAG | %d | %.1f%% | %.3f |" % (H1["n"], p["acc_mean"] * 100, p["acc_std"]))
    L.append("| 规则增强 RAG | %d | %.1f%% | %.3f |" % (H1["n"], r["acc_mean"] * 100, r["acc_std"]))
    L.append("")
    L.append("## 6.2 分场景增益（H2）")
    L.append("| 场景 | 样本量 | 通用LLM | 纯文本RAG | 规则增强RAG | 规则增益 |")
    L.append("|------|-----|-----|-----|-----|-----|")
    name = {"compliance": "合规", "marketing": "营销", "product": "选品"}
    for sc, d in H2["by_scenario"].items():
        L.append("| %s | %d | %.1f%% | %.1f%% | %.1f%% | %+.1f%% |" % (
            name.get(sc, sc), d["n"], d["general_llm"] * 100, d["plain_rag"] * 100, d["rule_rag"] * 100, d["gain_vs_general"] * 100))
    if H2["anova"] and H2["anova"].get("F") is not None:
        a = H2["anova"]
        L.append("")
        L.append("单因素方差分析：F=%.2f, %s（分组：%s）。" % (a["F"], pfmt(a["p"]), ", ".join(a["groups"])))
    L.append("")
    L.append("## 6.3 可解释性（H3）")
    L.append("| 条件 | 中位 | 均值 | 标准差 |")
    L.append("|------|-----|-----|-----|")
    for k, lab in [("general_llm", "通用LLM"), ("plain_rag", "纯文本RAG"), ("rule_rag", "规则增强RAG")]:
        d = H3[k]
        L.append("| %s | %.2f | %.2f | %.3f |" % (lab, d["median"], d["mean"], d["std"]))
    w = H3["wilcoxon_rule_vs_general"]
    if w and "W" in w:
        L.append("")
        L.append("Wilcoxon 符号秩检验：W=%.1f, Z=%s, %s。" % (w["W"], w["z"], pfmt(w["p"])))
    L.append("")
    L.append("## 6.5 问题类型 × 方法（H5）")
    L.append("| 问题类型 | 样本量 | 通用LLM | 纯文本RAG | 规则增强RAG | 规则增益 |")
    L.append("|------|-----|-----|-----|-----|-----|")
    tn = {"fact": "事实型", "compare": "比较型", "reasoning": "推理型"}
    for qt, d in H5["by_type"].items():
        L.append("| %s | %d | %.1f%% | %.1f%% | %.1f%% | %+.1f%% |" % (
            tn.get(qt, qt), d["n"], d["general_llm"] * 100, d["plain_rag"] * 100, d["rule_rag"] * 100, d["rule_gain"] * 100))
    twa = H5["two_way_anova"]
    if isinstance(twa, dict) and "C(qtype):C(method)" in twa:
        it = twa["C(qtype):C(method)"]
        L.append("")
        L.append("双因素方差分析交互项：F=%s, %s。" % (it["F"], pfmt(it["p"])))
    L.append("")
    L.append("## 6.6 错误分析（按实际结果统计）")
    L.append("| 错误类型 | 数量 | 占比 |")
    L.append("|------|-----|-----|")
    en = {"correct_or_partial": "正确/部分正确", "retrieval_fail": "检索失败", "rule_missing": "规则缺失",
          "hallucination": "生成幻觉", "annotation_dispute": "标注争议"}
    for k, d in stats["error_analysis"].items():
        L.append("| %s | %d | %.1f%% |" % (en.get(k, k), d["count"], d["pct"]))
    return "\n".join(L)


def check_deps(strict):
    # 本脚本只用 Python 标准库即可运行：统计用纯 Python 实现（purestats.py），
    # API 调用用标准库 urllib。scipy/statsmodels/pandas 只是可选加速/交叉验证。
    optional = []
    for m in ["scipy", "statsmodels", "pandas"]:
        try:
            __import__(m)
        except Exception:
            optional.append(m)
    if optional:
        print("[提示] 未安装可选库: " + ", ".join(optional) +
              "\n        —— 不影响运行，统计将用内置纯 Python 实现计算（结果等价）。\n")
    return optional


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scenario", type=int, default=None)
    ap.add_argument("--max-q", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-judge", action="store_true", help="关闭 LLM 评委打分，改用免费的关键点覆盖率")
    args = ap.parse_args()
    do_judge = (not args.dry_run) and (not args.no_judge)
    check_deps(strict=not args.dry_run)

    dataset = get_evaluation_dataset()
    print("评测集总量：%d 条" % len(dataset), get_dataset_stats())
    subset = select_questions(dataset, args.per_scenario, args.max_q)
    print("本次运行：%d 条\n" % len(subset))
    if do_judge:
        print("[评分口径] LLM 评委打分（更可信）。预计 API 调用≈%d 次（每题6次：生成3+评分3）。" % (len(subset) * 6))
        print("           如需省钱，可在命令后加 --no-judge 改用免费覆盖率打分。\n")
    else:
        print("[评分口径] 关键点覆盖率（免费，无需额外 API）。\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.dry_run:
        print("[DRY-RUN] 使用回退模板，仅自检脚本与统计管线，输出带 _DRYRUN 标记，不可用于论文。")
        llm = LLMClient()  # 无 key -> 模板
        outdir = os.path.join(RESULTS_DIR, "_dryrun")
        os.makedirs(outdir, exist_ok=True)
        suffix = "_DRYRUN"
    else:
        llm = StrictLLMClient()
        preflight(llm)
        outdir = RESULTS_DIR
        suffix = ""

    progress_path = None if args.dry_run else os.path.join(outdir, "progress_records.csv")
    if progress_path:
        print("[断点续跑] 进度文件：%s" % progress_path)
        print("           每完成 1 题自动保存；若中断，下次双击会从这里继续。\n")
    records = run(subset, llm, judge=do_judge, dry=args.dry_run, progress_path=progress_path)
    stats = compute_stats(records)
    meta = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            "scoring": ("LLM评委(0-100归一)" if do_judge else "关键点覆盖率"),
            "api_available": (not args.dry_run) and llm.is_available()}
    stats["_meta"] = meta

    audit = os.path.join(outdir, "audit_per_question%s.csv" % suffix)
    save_audit(records, audit)
    with open(os.path.join(outdir, "thesis_stats%s.json" % suffix), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    ch = chapter(stats, meta)
    with open(os.path.join(outdir, "thesis_chapter_6_results%s.md" % suffix), "w", encoding="utf-8") as f:
        f.write(ch)

    print("\n" + ch)
    print("\n[OK] 逐题留痕: %s" % audit)
    print("[OK] 统计JSON / 第六章 已写入: %s" % outdir)
    if args.dry_run:
        print("\n[!] 这是 DRY-RUN 结果，数字来自模板兜底，绝不能写进论文。请配置 API Key 后重跑。")


if __name__ == "__main__":
    main()
