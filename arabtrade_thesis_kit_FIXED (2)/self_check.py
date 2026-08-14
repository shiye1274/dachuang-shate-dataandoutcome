# -*- coding: utf-8 -*-
"""
self_check.py —— 自检：不联网、不调用 API、不花钱，验证这套程序是否可信。
逐项打印 [OK] / [FAIL]，最后给总结。看到全是 [OK] 就说明统计与数据管线正确。
"""
import os, sys, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL = [], []
def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(("  [OK] " if ok else "  [FAIL] ") + name + (("  -> " + detail) if detail else ""))

def have(m):
    return importlib.util.find_spec(m) is not None

print("=" * 56)
print("  自检开始（不联网 / 不调用 API / 不花钱）")
print("=" * 56)

print("\n[1] 运行环境")
print("    Python:", sys.version.split()[0])
for m in ["scipy", "statsmodels", "pandas", "numpy"]:
    print("    %s: %s" % (m, "已安装" if have(m) else "未安装(用纯Python等价实现)"))

print("\n[2] 统计函数对拍『教科书已知值』（不依赖任何库）")
import purestats as PS
check("正态双侧 z=1.96 -> p≈0.05", abs(2 * PS.norm_sf(1.96) - 0.05) < 1e-3, "得 %.5f" % (2 * PS.norm_sf(1.96)))
check("betai(0.5,0.5,0.5)=0.5", abs(PS.betai(0.5, 0.5, 0.5) - 0.5) < 1e-6, "得 %.5f" % PS.betai(0.5, 0.5, 0.5))
t, _ = PS.ttest_rel([2, 4, 6, 8, 10], [1, 2, 3, 4, 5])
check("配对t检验 手算=4.2426", abs(t - 4.2426) < 1e-3, "得 %.4f" % t)
Fa, _ = PS.f_oneway([1, 2, 3, 4], [3, 4, 5, 6])
ta, _ = PS.ttest_ind([1, 2, 3, 4], [3, 4, 5, 6])
check("单因素ANOVA F = 两样本t检验的平方(数学恒等式)", abs(Fa - ta * ta) < 1e-6, "F=%.4f, t²=%.4f" % (Fa, ta * ta))

print("\n[3] 统计函数对拍『scipy 黄金标准』")
if have("scipy"):
    from scipy import stats as SS
    A = [5, 6, 7, 8, 9, 7, 6, 8, 9, 10]
    B = [4, 5, 5, 6, 7, 6, 5, 7, 8, 8]
    t1, p1 = PS.ttest_rel(A, B)
    t2, p2 = SS.ttest_rel(A, B)
    check("配对t检验 与 scipy 一致", abs(t1 - t2) < 1e-6 and abs(p1 - p2) < 1e-6, "纯Python t=%.5f / scipy t=%.5f" % (t1, t2))
    g = [[1, 2, 3, 4], [2, 3, 4, 5], [5, 6, 7, 8]]
    f1, fp1 = PS.f_oneway(*g)
    f2, fp2 = SS.f_oneway(*g)
    check("单因素ANOVA 与 scipy 一致", abs(f1 - f2) < 1e-6 and abs(fp1 - fp2) < 1e-6, "纯Python F=%.5f / scipy F=%.5f" % (f1, f2))
else:
    print("    (本机未装 scipy，跳过；你的电脑若装了 scipy 会自动逐项对比。)")

print("\n[4] 评测集完整性")
from evaluation_dataset import get_evaluation_dataset, get_dataset_stats
ds = get_evaluation_dataset()
check("题目总数 = 833", len(ds) == 833, "实际 %d" % len(ds))
empty_q = sum(1 for it in ds if not str(it.get("question", "")).strip())
empty_a = sum(1 for it in ds if not str(it.get("standard_answer", "")).strip())
check("无空问题", empty_q == 0, "空问题 %d 条" % empty_q)
check("无空标准答案", empty_a == 0, "空标准答案 %d 条" % empty_a)
types = set(it["type"] for it in ds)
scs = set(it["scenario"] for it in ds)
check("题型仅含 fact/compare/reasoning", types <= {"fact", "compare", "reasoning"}, str(sorted(types)))
check("场景仅含 compliance/marketing/product", scs <= {"compliance", "marketing", "product"}, str(sorted(scs)))
print("    分布:", get_dataset_stats())

print("\n[5] 统计管线端到端（合成数据，不调用 API）")
import run_thesis_experiment as R
scen = ["compliance", "marketing", "product"]
typ = ["fact", "compare", "reasoning"]
recs = []
for i in range(30):
    base = (i % 5) / 10.0
    recs.append({
        "id": "T%03d" % i, "question": "q", "standard_answer": "s",
        "type": typ[i % 3], "scenario": scen[i % 3], "source": "",
        "retrieved_count": i % 6, "rule_hit": (i % 2 == 0),
        "gen_answer": "a", "plain_answer": "b", "rule_answer": "## 合规提醒 建议 风险 来源",
        "gen_score": base, "plain_score": base + 0.05, "rule_score": base + 0.10,
        "gen_interp": 0.25, "plain_interp": 0.5, "rule_interp": 0.75,
    })
try:
    stats = R.compute_stats(recs)
    ok = (stats["H1"]["n"] == 30 and stats["total_questions"] == 30 and all(k in stats for k in ["H1", "H2", "H3", "H5", "error_analysis"]))
    check("compute_stats 正常产出 H1/H2/H3/H5", ok)
    prog = stats["H1"]["rule_rag"]["acc_mean"]
    hand = round(sum(r["rule_score"] for r in recs) / len(recs), 4)
    check("H1 规则组均值与手算一致", abs(prog - hand) < 1e-4, "程序 %.4f / 手算 %.4f" % (prog, hand))
except Exception as e:
    check("compute_stats 正常产出", False, str(e))

print("\n[6] 打分函数行为")
from thesis_results import compute_factual_accuracy, compute_four_segment_score
same = compute_factual_accuracy("沙特VAT标准税率15%，2020年7月起上调。", "沙特VAT标准税率15%")
miss = compute_factual_accuracy("今天天气不错。", "沙特VAT标准税率15%，SABER认证，PC证书")
seg = compute_four_segment_score("## 合规提醒 xxx 建议 xxx 风险 xxx 来源 PDPL")
check("答案含标准答案关键词 -> 高分", same >= 0.5, "得 %.2f" % same)
check("答非所问 -> 低分", miss <= 0.2, "得 %.2f" % miss)
check("四段式结构 -> 结构分高", seg >= 0.75, "得 %.2f" % seg)

print("\n" + "=" * 56)
if FAIL:
    print("  自检结果：发现 %d 项不通过 [X] —— 请把本窗口截图发给小白。" % len(FAIL))
    print("  未通过：", "; ".join(FAIL))
    print("=" * 56)
    sys.exit(1)
else:
    print("  自检结果：全部 %d 项通过 [OK] —— 统计与数据管线可信。" % len(PASS))
    print("  说明：统计结果与教科书 / scipy / 手算三方对得上；")
    print("        每题打分都会写进 audit_per_question.csv，可逐条回溯。")
    print("=" * 56)
