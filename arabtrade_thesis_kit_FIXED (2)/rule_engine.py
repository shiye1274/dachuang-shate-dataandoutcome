"""
规则层引擎 — Rule-enhanced compliance engine for Saudi apparel exports

将不可违反的硬约束从生成中剥离，以显式规则 + 校验器实现。
当问题触及合规、税务、标签、敏感表达时，优先由规则给出确定性判断或强制提示。
"""
from typing import Optional, Dict, List, Any
import re


class RuleResult:
    """规则匹配结果"""
    def __init__(self, triggered: bool, rule_id: str = "", rule_name: str = "",
                 judgment: str = "", suggestion: str = "", risk: str = "",
                 source: str = "", priority: str = "high"):
        self.triggered = triggered
        self.rule_id = rule_id
        self.rule_name = rule_name
        self.judgment = judgment
        self.suggestion = suggestion
        self.risk = risk
        self.source = source
        self.priority = priority  # high / medium / low

    def to_dict(self):
        return {
            "triggered": self.triggered,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "judgment": self.judgment,
            "suggestion": self.suggestion,
            "risk": self.risk,
            "source": self.source,
            "priority": self.priority,
        }


class RuleEngine:
    """
    规则层引擎。
    每条规则包含：trigger_conditions → judge → suggest → cite
    """

    def __init__(self):
        self.rules = self._init_rules()

    def _init_rules(self):
        return [
            self._rule_r1_export_cert,
            self._rule_r2_label,
            self._rule_r3_sensitive_expression,
            self._rule_r4_vat_tax,
            self._rule_r5_pdpl,
            self._rule_r6_prohibited_items,
            self._rule_r7_image_modesty,
            self._rule_r8_packaging,
        ]

    # ========== R1: 出口认证与清关 ==========
    def _rule_r1_export_cert(self, question: str, context: Optional[Dict] = None) -> RuleResult:
        triggers = [
            r"(清关|customs|clearance|发货|ship|SABER|SC证书|PC证书|认证|certif)",
            r"(出口|export|报关|退运|到港|清关文件)",
            r"(纺织|服装|clothing|garment|apparel|textile|服饰)",
        ]
        if not any(re.search(t, question, re.IGNORECASE) for t in triggers):
            return RuleResult(triggered=False)

        return RuleResult(
            triggered=True,
            rule_id="R1",
            rule_name="出口认证与清关",
            judgment="该品类受沙特纺织品技术法规约束，出口前须在 SABER 平台完成 PC 注册并逐票申领 SC；无 SC 不予清关，面临退运。",
            suggestion="1) 在 SABER 平台注册企业账号；2) 提交产品资料申请 PC（产品证书，有效期 1 年）；3) 每批货物到港前申请 SC（装运证书）；4) 预留 2-4 周认证周期。",
            risk="自 2025年1月1日起 SC 已全面前置，到港前未取得 SC 将直接退运，损失巨大。",
            source="SASO《纺织产品技术法规》(2019-12)；SALEEM/SABER 合格评定体系",
        )

    # ========== R2: 标签与成分标注 ==========
    def _rule_r2_label(self, question: str, context: Optional[Dict] = None) -> RuleResult:
        triggers = [
            r"(标签|label|吊牌|tag|成分|纤维|fabric content|composition)",
            r"(标注|mark|标识|洗涤|care|原产国|origin|made in)",
            r"(尺码|size|面料|material面料)",
        ]
        if not any(re.search(t, question, re.IGNORECASE) for t in triggers):
            return RuleResult(triggered=False)

        return RuleResult(
            triggered=True,
            rule_id="R2",
            rule_name="标签与成分标注",
            judgment="标签须用阿语或阿英双语，完整标注纤维成分百分比（从高到低）、原产国、生产商/在沙经销商信息与洗涤养护符号；不得使用误导性措辞。",
            suggestion="1) 使用阿英双语标签；2) 标注纤维成分及百分比（如「95%棉, 5%氨纶」）；3) 标注「Made in China / صنع في الصين」；4) 标注进口商/经销商信息；5) 添加 ISO 洗涤养护符号。",
            risk="缺少阿语标签、成分未标注或使用误导性措辞可能导致海关暂扣、罚款或退运。",
            source="SASO《纺织产品技术法规》标签条款；SASO 1173；SASO ISO 3758",
        )

    # ========== R3: 图样与敏感表达 ==========
    def _rule_r3_sensitive_expression(self, question: str, context: Optional[Dict] = None) -> RuleResult:
        triggers = [
            r"(图样|图案|print|design|花纹|symbol|人物|模特|model|图像)",
            r"(敏感|sensitive|cultural|宗教|islam|muslim|清真|halal)",
            r"(国徽|国旗|emblem|flag|cross|cross十字架)",
            r"(酒精|alcohol|wine|猪|pig|pork|暴露|revealing|sexy|性感)",
            r"(文案|copy|广告|advert|营销|marketing|表达)",
        ]
        if not any(re.search(t, question, re.IGNORECASE) for t in triggers):
            return RuleResult(triggered=False)

        return RuleResult(
            triggered=True,
            rule_id="R3",
            rule_name="图样与敏感表达",
            judgment="包装与素材不得使用沙特国徽（椰枣树+交叉双剑）；禁止酒精、猪、色情或不雅呈现；女性着装须符合当地 modest 规范；禁含十字架及其他宗教符号的不当使用。",
            suggestion="1) 女性模特着装保守（遮盖身体曲线）；2) 不使用沙特国徽/国旗；3) 删除含酒精/猪/暴露元素的设计；4) 避免含十字架等宗教符号；5) 文案避免政治敏感话题。",
            risk="违规可导致产品下架、平台封号、品牌声誉严重受损，严重时面临法律诉讼。",
            source="沙特商务部国徽使用禁令(2026)；沙特海关禁止/限制进口清单；Cader(2015)",
        )

    # ========== R4: 增值税与税务 ==========
    def _rule_r4_vat_tax(self, question: str, context: Optional[Dict] = None) -> RuleResult:
        triggers = [
            r"(增值税|VAT|tax|税务|税|ZATCA|FATOORA|发票|invoice)",
            r"(申报|file|注册|register|电子发票|e-invoice)",
            r"(罚款|penalty|fine|SAR)",
        ]
        if not any(re.search(t, question, re.IGNORECASE) for t in triggers):
            return RuleResult(triggered=False)

        return RuleResult(
            triggered=True,
            rule_id="R4",
            rule_name="增值税与电子发票",
            judgment="沙特 VAT 标准税率 15%；跨境电商年销售额超 SAR 375,000 须注册 VAT；须使用 FATOORA 规范电子发票（含 QR 码、增值税号）。",
            suggestion="1) 评估是否触发 VAT 注册义务（年销售额 SAR 375,000 门槛）；2) 在 ZATCA 注册 VAT 号；3) 部署 FATOORA 兼容的电子发票系统；4) 按期申报 VAT。",
            risk="未注册罚款最高 SAR 10,000；未申报罚款应缴税款的 5%-25%；发票不合规每张罚 SAR 50-100。",
            source="ZATCA 增值税指南；ZATCA FATOORA 规范",
        )

    # ========== R5: PDPL 数据合规 ==========
    def _rule_r5_pdpl(self, question: str, context: Optional[Dict] = None) -> RuleResult:
        triggers = [
            r"(PDPL|数据|data|隐私|privacy|个人信息|personal info|GDPR)",
            r"(合规|跨境|transfer|用户权利|用户数据|customer data)",
            r"(收集|collect|处理|process|同意|consent)",
        ]
        if not any(re.search(t, question, re.IGNORECASE) for t in triggers):
            return RuleResult(triggered=False)

        return RuleResult(
            triggered=True,
            rule_id="R5",
            rule_name="PDPL 数据合规",
            judgment="PDPL 要求数据控制者取得数据主体明确同意、披露数据用途、保障数据安全。跨境传输受限，须经 SDAIA 评估。",
            suggestion="1) 在电商网站/App 公布阿语隐私政策；2) 取得用户数据收集的明确同意（checkbox）；3) 披露数据用途和共享范围；4) 评估跨境数据传输必要性；5) 指定数据保护负责人。",
            risk="违规罚款最高 SAR 5,000,000（约 133 万美元）；约 70% 在沙电商网站 PDPL 合规存在缺陷。",
            source="SDAIA PDPL 官方文本及实施条例；Alashwali & Alhuzali (2026)",
        )

    # ========== R6: 禁止进口品类 ==========
    def _rule_r6_prohibited_items(self, question: str, context: Optional[Dict] = None) -> RuleResult:
        triggers = [
            r"(禁止|prohibit|banned|forbidden|限制|restrict)",
            r"(进口|import|入境|customs)",
            r"(宗教|religious|政治|political|赌博|gambling|色情|porn)",
        ]
        if not any(re.search(t, question, re.IGNORECASE) for t in triggers):
            return RuleResult(triggered=False)

        return RuleResult(
            triggered=True,
            rule_id="R6",
            rule_name="禁止进口品类",
            judgment="沙特禁止进口含宗教标志不当使用、酒精/猪相关图案、政治敏感内容、赌博/色情元素的服装服饰。",
            suggestion="1) 出运前逐项审核产品图样和文案；2) 对照沙特海关禁止/限制进口清单排查；3) 有疑虑的产品更换设计或删除元素。",
            risk="含禁止元素的货物将被海关扣押并销毁，严重违规可能影响出口企业信用记录。",
            source="沙特海关禁止/限制进口清单",
        )

    # ========== R7: 模特与视觉规范 ==========
    def _rule_r7_image_modesty(self, question: str, context: Optional[Dict] = None) -> RuleResult:
        triggers = [
            r"(模特|model|拍摄|photo|shoot|图片|image|视觉|visual)",
            r"(穿衣|dress|着装|outfit|wear|展示|display)",
            r"(产品图|listing|主图|详情图|商品图)",
        ]
        if not any(re.search(t, question, re.IGNORECASE) for t in triggers):
            return RuleResult(triggered=False)

        return RuleResult(
            triggered=True,
            rule_id="R7",
            rule_name="模特与视觉规范",
            judgment="女性模特须着装 modest（遮盖身体曲线，不暴露），建议戴头巾（hijab）；姿势端庄，避免舞蹈或暗示性姿势；可单独展示产品以降低文化风险。",
            suggestion="1) 女性模特穿长袖宽松服装 + 头巾；2) 姿势自然端庄；3) 可添加无模特平铺图选项；4) 背景简洁/中性色；5) 展示面料细节和尺码对比。",
            risk="暴露的女性形象可能引发文化争议、社交媒体抵制、平台强制下架。",
            source="Amazon.sa/Noon.com 图片规范；沙特商务部广告规范",
        )

    # ========== R8: 包装规范 ==========
    def _rule_r8_packaging(self, question: str, context: Optional[Dict] = None) -> RuleResult:
        triggers = [
            r"(包装|packag|包裝|礼盒|gift box|袋|bag)",
            r"(品牌|brand|logo|corporate identity|VI)",
            r"(设计|design|外观|appearance)",
        ]
        if not any(re.search(t, question, re.IGNORECASE) for t in triggers):
            return RuleResult(triggered=False)

        return RuleResult(
            triggered=True,
            rule_id="R8",
            rule_name="包装规范",
            judgment="包装避免使用人物图像（尤其是女性形象），禁用沙特国徽；宜采用金色、深红、墨绿等高贵色调配以阿拉伯风格花纹/几何图案。",
            suggestion="1) 包装图案避免人物/动物形象；2) 不使用沙特国徽及交叉双剑；3) 推荐使用几何/植物花纹；4) 深色/金色系更受欢迎；5) 阿语文字右对齐。",
            risk="人物图案或国徽使用可能被海关扣押，或引发文化冒犯。",
            source="沙特商务部包装规则；中东包装设计最佳实践",
        )

    # ========== Main interface ==========
    def evaluate(self, question: str, context: Optional[Dict] = None) -> List[RuleResult]:
        """对问题逐条运行规则，返回所有触发的规则结果"""
        results = []
        for rule_fn in self.rules:
            result = rule_fn(question, context)
            if result.triggered:
                results.append(result)
        return results

    def has_critical_hit(self, question: str, context: Optional[Dict] = None) -> bool:
        """是否有任何规则被触发"""
        results = self.evaluate(question, context)
        return len(results) > 0

    def get_four_segment_output(self, question: str, context: Optional[Dict] = None) -> Dict:
        """
        四段式可解释输出:
        1. 合规提醒 — 触发的规则硬约束
        2. 行动建议 — 可执行步骤
        3. 风险提示 — 潜在风险
        4. 引用来源 — 知识库出处
        """
        results = self.evaluate(question, context)
        
        if not results:
            return {
                "has_rule_hit": False,
                "compliance_reminder": "未检测到硬约束触发，可转入 RAG 检索生成。",
                "action_suggestions": [],
                "risk_alerts": [],
                "citations": [],
            }

        compliance_parts = []
        action_parts = []
        risk_parts = []
        source_parts = []

        for r in results:
            compliance_parts.append(f"【{r.rule_id} {r.rule_name}】{r.judgment}")
            action_parts.append(r.suggestion)
            risk_parts.append(f"⚠️ {r.risk}")
            source_parts.append(f"📖 {r.source}")

        return {
            "has_rule_hit": True,
            "compliance_reminder": "\n\n".join(compliance_parts),
            "action_suggestions": action_parts,
            "risk_alerts": risk_parts,
            "citations": source_parts,
        }


def create_rule_engine() -> RuleEngine:
    return RuleEngine()
