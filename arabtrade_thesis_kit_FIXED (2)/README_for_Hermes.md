> 【重要更新】本版本已改为“零依赖”：统计用纯 Python 实现（purestats.py），API 调用用标准库 urllib。
> 不需要 pip install scipy/statsmodels/pandas/requests，任何 Python 3 直接运行： python run_thesis_experiment.py
> （若已装 scipy 会自动用它交叉验证，但不装也能算出等价结果。）

# 给 Hermes 的运行说明（论文实验，防幻觉版）

本目录新增两个脚本，用来产出论文第六章需要的**真实**实验数字与图。
所有数字都由脚本从数据算出，**没有任何写死或编造**。

## 0. 前置：装依赖 + 配 API Key
```bash
pip install requests scipy statsmodels pandas
export DEEPSEEK_API_KEY=sk-你的key
# 可选：export DEEPSEEK_MODEL=deepseek-chat
```

## 1. 跑完整实验：run_thesis_experiment.py
```bash
python run_thesis_experiment.py                 # 跑全量 833 条
python run_thesis_experiment.py --per-scenario 100   # 每场景各100条(更快)
python run_thesis_experiment.py --max-q 150     # 只跑前150条
python run_thesis_experiment.py --dry-run       # 不连API，仅自检脚本
```
产出（写到 results/）：
- `thesis_chapter_6_results.md`——可直接对照修改论文第六章的表格与结论
- `thesis_stats.json`——所有统计量（含 H1 的 t、H2 的 F、H3 的 Z/W、H5 双因素 ANOVA）
- `audit_per_question.csv`——**逐题留痕**：每题三种方法的原始回答 + 打分，任一数字都能回溯核对

### 这版脚本相比旧 thesis_results.py 修好了什么（重要）
1. **删掉了 30 条上限**（旧脚本第388行 `max_q=min(30,...)`），默认跑全量、按场景分层。
2. **连不上 API 直接报错退出**，绝不静默回退成模板假数据（旧脚本会悄悄兜底）。
3. **真实计算显著性检验**：旧脚本第六章把 `F=`、`Z=`、`p<0.05` 写死/留空；本版用 scipy/statsmodels 实算：
   - H1：配对 t 检验（rule vs general、rule vs plain）+ Cohen's d
   - H2：三场景增益单因素 ANOVA + Tukey 事后比较
   - H3：Wilcoxon 符号秩检验（含 Z 近似）
   - H5：问题类型 × 方法 双因素 ANOVA（含交互项）
4. **6.6 错误分析按实际结果统计**（旧脚本是写死的 8/5/4/2/11）。
5. 开跑前先检查依赖，缺库立即提示，**不会白白消耗 API 调用**。

## 2. 生成四段式输出样例图：make_sample_figure.py
```bash
python make_sample_figure.py                       # 默认用一个合规问题
python make_sample_figure.py "斋月女装营销要注意什么？"   # 换成你的问题
```
产出：`results/sample_output.md`、`sample_output.html`、`sample_output.png`
- 命中规则时四段内容全部来自规则引擎，确定性、可复现、无需 API。
- 用浏览器打开 html 截图，或直接用 png，作为论文 §4.5 的“四段式回答样例”Figure。

## 3. 跑完后要回填到论文的地方
- §4.3 / §5.2 的评测集口径：现在实际是 **833 条**（事实470/推理197/比较166），把旧的“299条/47推理”改掉。
- 第六章 6.1–6.6 全部表格与检验统计量：用 thesis_chapter_6_results.md 覆盖。
- §4.5：插入 sample_output.png。
- 知识库规模：脚本实测 **7508 条**（README 写的 62 是旧数字）。

## 4. 自查清单（防幻觉）
- [ ] thesis_stats.json 里 `_meta.api_available` 必须是 `true`
- [ ] `_scipy_available` 必须是 `true`（否则统计量为空）
- [ ] total_questions 是你预期的样本量
- [ ] 论文里每个数字都能在 audit_per_question.csv 找到出处
