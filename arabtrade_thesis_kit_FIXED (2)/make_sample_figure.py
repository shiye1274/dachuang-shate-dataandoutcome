#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_sample_figure.py — 生成「1 个完整四段式输出样例」用于论文图（§4.5）

- 对一个真实合规问题，用规则引擎 + 知识库生成四段式输出（合规提醒/行动建议/风险提示/引用来源）。
- 规则命中时四段内容完全来自规则引擎，确定性、无需 API（真实可复现）。
- 输出 sample_output.md 与 sample_output.html；用浏览器打开 html 截图即可作为 Figure。

用法： python make_sample_figure.py ["你的问题"]
"""
import os, sys, html, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_runner import KnowledgeBase
from rule_engine import create_rule_engine

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def build(question):
    kb = KnowledgeBase()
    eng = create_rule_engine()
    ro = eng.get_four_segment_output(question)
    retrieved = kb.search(question, top_k=5)
    segs = []
    segs.append(("合规提醒", ro.get("compliance_reminder", "未检测到硬约束")))
    segs.append(("行动建议", "\n".join("- " + s for s in ro.get("action_suggestions", [])[:4]) or "- （无）"))
    segs.append(("风险提示", "\n".join("- " + s for s in ro.get("risk_alerts", [])[:4]) or "- （无）"))
    cites = ro.get("citations", [])[:4] + [r.get("source", "") for r in retrieved[:3] if r.get("source")]
    segs.append(("引用来源", "\n".join("- " + c for c in dict.fromkeys([c for c in cites if c])) or "- （无）"))
    return ro.get("has_rule_hit", False), segs


def to_md(question, hit, segs):
    L = ["# 图 X　规则增强 RAG 四段式输出样例", "",
         "**用户问题：**" + question, "",
         "**规则是否命中：**" + ("是" if hit else "否（降级为 RAG 四段式）"), ""]
    for t, c in segs:
        L.append("## " + t)
        L.append(c)
        L.append("")
    return "\n".join(L)


def to_html(question, hit, segs):
    css = "body{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;background:#f6f7f9;padding:32px;}" \
          ".card{max-width:760px;margin:auto;background:#fff;border-radius:14px;box-shadow:0 4px 24px rgba(0,0,0,.08);overflow:hidden;}" \
          ".q{background:#2f6df6;color:#fff;padding:18px 24px;font-size:16px;}" \
          ".hit{font-size:12px;opacity:.85;margin-top:6px;}" \
          ".seg{padding:16px 24px;border-top:1px solid #eef0f3;}" \
          ".seg h3{margin:0 0 8px;font-size:15px;}" \
          ".s0 h3{color:#d9480f;}.s1 h3{color:#1971c2;}.s2 h3{color:#e8590c;}.s3 h3{color:#2b8a3e;}" \
          ".seg pre{white-space:pre-wrap;margin:0;font-family:inherit;font-size:14px;color:#333;line-height:1.7;}"
    parts = ['<!doctype html><meta charset="utf-8"><style>%s</style>' % css,
             '<div class="card"><div class="q">🧕 用户问题：%s<div class="hit">规则命中：%s</div></div>' % (
                 html.escape(question), "是" if hit else "否")]
    for i, (t, c) in enumerate(segs):
        parts.append('<div class="seg s%d"><h3>%s</h3><pre>%s</pre></div>' % (i, html.escape(t), html.escape(c)))
    parts.append("</div>")
    return "".join(parts)


def main():
    q = sys.argv[1] if len(sys.argv) > 1 else "第一次往沙特发女装，需要做哪些合规准备？"
    hit, segs = build(q)
    os.makedirs(RESULTS, exist_ok=True)
    md = to_md(q, hit, segs)
    open(os.path.join(RESULTS, "sample_output.md"), "w", encoding="utf-8").write(md)
    open(os.path.join(RESULTS, "sample_output.html"), "w", encoding="utf-8").write(to_html(q, hit, segs))
    print(md)
    print("\n[OK] 已生成 results/sample_output.md 和 results/sample_output.html")
    print("[提示] 用浏览器打开 sample_output.html 截图，即可作为论文 Figure（四段式回答图）。")


if __name__ == "__main__":
    main()
