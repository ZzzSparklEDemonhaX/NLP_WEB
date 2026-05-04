from __future__ import annotations

import html
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import jieba
import jieba.posseg as pseg
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from opencc import OpenCC
except ImportError:  # pragma: no cover - the UI still works without OpenCC.
    OpenCC = None


DEFAULT_TEXTS = {
    "新闻舆情": "新华社北京5月3日电，人工智能技术正在加速进入教育、医疗和城市治理等领域。专家表示，自然语言处理能够帮助系统理解群众诉求，并提升公共服务效率。",
    "电商评论": "这款无线耳机音质很清晰，降噪效果不错，物流速度也很快。但是续航时间比宣传的略短，客服回复比较及时。",
    "课程论文": "自然语言处理研究计算机与人类语言之间的交互，词法分析通常包括文本规范化、中文分词、词性标注和关键词统计等基础任务。",
}

BASE_DICTIONARY = {
    "新华社",
    "北京",
    "人工智能",
    "自然语言处理",
    "技术",
    "教育",
    "医疗",
    "领域",
    "城市治理",
    "专家",
    "系统",
    "群众",
    "诉求",
    "公共服务",
    "效率",
    "无线耳机",
    "音质",
    "清晰",
    "降噪",
    "效果",
    "不错",
    "物流",
    "速度",
    "续航时间",
    "宣传",
    "客服",
    "回复",
    "及时",
    "研究",
    "计算机",
    "人类语言",
    "之间",
    "交互",
    "词法分析",
    "文本规范化",
    "中文分词",
    "词性标注",
    "关键词",
    "统计",
    "基础任务",
    "加速",
    "进入",
    "改变",
}

STOPWORDS = {
    "的",
    "了",
    "和",
    "与",
    "等",
    "在",
    "也",
    "很",
    "比",
    "并",
    "能够",
    "正在",
    "通常",
    "包括",
}

TAG_META = {
    "n": ("名词", "#ef4444"),
    "v": ("动词", "#2563eb"),
    "a": ("形容词", "#16a34a"),
    "r": ("代词", "#9333ea"),
    "m": ("数词", "#ea580c"),
    "t": ("时间词", "#0f766e"),
    "eng": ("英文", "#64748b"),
    "x": ("其他", "#64748b"),
}

POS_LEXICON = {
    "自然语言处理": "n",
    "人工智能": "n",
    "词法分析": "n",
    "文本规范化": "n",
    "中文分词": "n",
    "词性标注": "n",
    "人类语言": "n",
    "计算机": "n",
    "无线耳机": "n",
    "音质": "n",
    "物流": "n",
    "客服": "n",
    "研究": "v",
    "进入": "v",
    "加速": "v",
    "改变": "v",
    "提升": "v",
    "帮助": "v",
    "理解": "v",
    "分析": "v",
    "生成": "v",
    "清晰": "a",
    "不错": "a",
    "及时": "a",
    "基础": "a",
}

SEGMENTER_DETAILS = {
    "FMM 正向最大匹配": {
        "伪代码": """从句首开始：
1. 取当前位置起最长候选串
2. 若候选串在词典中，则输出该词
3. 否则候选串长度减 1
4. 移动到下一个未切分位置，直到文本结束""",
        "核心逻辑": "从左到右做词典贪心匹配，优先选择最长词。算法完全依赖词典，路径清晰，适合课堂展示词典覆盖率对分词结果的影响。",
        "公式指标": "匹配目标：w* = argmax |w|，其中 w 属于词典 D，且 w 是当前位置起始的前缀。评测常用边界 Precision / Recall / F1。",
        "效果特点": "速度快、可解释性强；但遇到歧义时容易被句首方向误导，对未登录词识别能力弱。",
        "使用情景": "适合专有词典较稳定的场景，例如课程术语、产品名、机构名初步切分，也适合作为词典 baseline。",
    },
    "RMM 逆向最大匹配": {
        "伪代码": """从句尾开始：
1. 取当前位置向左的最长候选串
2. 若候选串在词典中，则输出该词
3. 否则候选串长度减 1
4. 继续向左扫描，最后反转输出顺序""",
        "核心逻辑": "从右到左做最大匹配。中文中不少歧义在逆向扫描时表现不同，因此 RMM 常与 FMM 对照观察。",
        "公式指标": "匹配目标：w* = argmax |w|，其中 w 属于词典 D，且 w 是当前位置结束的后缀。",
        "效果特点": "同样快速直观；在某些歧义句上优于 FMM，但仍受词典覆盖和最大词长限制。",
        "使用情景": "适合与 FMM 做方向性对比，展示“扫描方向会改变分词边界”的现象。",
    },
    "Bi-MM 双向最大匹配": {
        "伪代码": """同时运行 FMM 与 RMM：
1. 得到 forward_tokens
2. 得到 reverse_tokens
3. 选择词数更少的结果
4. 若词数相同，选择单字词更少的结果""",
        "核心逻辑": "把正向和逆向结果进行投票式选择，常用启发式是“词数少优先、单字词少优先”。",
        "公式指标": "score(S) = token_count(S) + λ * single_char_count(S)，选择 score 更小的切分。",
        "效果特点": "比单独 FMM/RMM 更稳，仍保持很强可解释性；但不能真正学习上下文概率。",
        "使用情景": "适合课堂 baseline 和轻量级工程场景，尤其适合讲解中文分词歧义消解。",
    },
    "Jieba 精确模式": {
        "伪代码": """1. 根据词典构建 DAG 候选词图
2. 对每个位置计算最大概率路径
3. 对未登录片段调用 HMM
4. 输出概率最优的分词序列""",
        "核心逻辑": "Jieba 结合前缀词典、DAG 动态规划和 HMM 未登录词识别，比纯词典贪心更能处理新词。",
        "公式指标": "路径目标：S* = argmax Σ log P(w_i)。HMM 使用状态序列 B/M/E/S 表示词首、词中、词尾、单字词。",
        "效果特点": "通用中文文本表现稳定，速度快，能识别一部分词典外词；但专业领域仍可能需要自定义词典。",
        "使用情景": "适合作为本应用默认算法，也适合新闻、评论、课程论文等通用中文文本。",
    },
    "Jieba 关闭 HMM": {
        "伪代码": """1. 只使用词典与 DAG
2. 不启用 HMM 新词识别
3. 按词典概率路径输出结果""",
        "核心逻辑": "关闭统计新词识别后，更容易观察词典本身对结果的影响，是 Jieba 精确模式的消融实验。",
        "公式指标": "路径仍基于词频概率，但不额外估计未登录词状态序列。",
        "效果特点": "结果更保守，专业词若不在词典中更容易被拆碎；可解释性更强。",
        "使用情景": "适合展示 HMM 的作用，也适合只信任领域词典、不希望模型自由猜新词的场景。",
    },
    "混合：Bi-MM + HMM": {
        "伪代码": """1. 先用 Bi-MM 得到粗切分
2. 对长且不在词典中的中文片段调用 Jieba HMM
3. 保留词典命中的稳定词
4. 合并输出最终 token""",
        "核心逻辑": "先用词典算法保证可解释性，再用 HMM 处理疑似未登录词，是规则与统计方法的折中方案。",
        "公式指标": "粗切分使用 Bi-MM score，未知片段使用 HMM 的 P(observation, state_sequence) 最大化。",
        "效果特点": "比纯词典更灵活，比完全统计方法更可控；效果取决于词典质量和未知片段判断规则。",
        "使用情景": "适合课程展示“规则 + 统计模型”的混合思路，也适合小项目中快速增强 baseline。",
    },
}

POS_TAGGER_DETAILS = {
    "Jieba HMM 统计模型": {
        "伪代码": """对每个当前分词 token：
1. 若命中课程词性词典，直接使用词典标签
2. 否则调用 Jieba 词性标注
3. 将细粒度标签归并为 n/v/a/r/m/t/eng/x
4. 输出与分词结果一一对应的词性序列""",
        "核心逻辑": "以 Jieba 的统计词性标注为主体，再用课程词典做后处理，保证展示中常见术语的标注更符合预期。",
        "公式指标": "准确率：Accuracy = 正确标注词数 / 可比词数。本页面 baseline 对比使用内置 gold set。",
        "效果特点": "通用文本表现较稳，对名词和动词识别较好；细粒度标签需要归并，专业术语最好配合词典修正。",
        "使用情景": "适合默认展示、新闻评论分析、课程论文摘要等场景，是当前页面推荐的词性标注方法。",
    },
    "规则 baseline": {
        "伪代码": """对每个当前分词 token：
1. 若命中词性词典，使用词典标签
2. 若匹配形容词后缀或形容词表，标为 a
3. 若匹配动词后缀或动词表，标为 v
4. 标点为 x，数字为 m，英文为 eng，其余默认 n""",
        "核心逻辑": "模拟 Brill Tagger 的思想：先给一个简单初始标签，再根据人工规则修正。规则少但非常透明。",
        "公式指标": "规则命中率和 Accuracy 都可以作为展示指标；本页面使用 Accuracy 与 Jieba HMM 对比。",
        "效果特点": "解释性最强、运行极快；但规则覆盖有限，遇到复杂上下文容易误标。",
        "使用情景": "适合作为 baseline、课堂讲解规则方法，也适合领域很窄且标签规则明确的小型系统。",
    },
}

GOLD_SEGMENTATION = [
    {
        "text": "自然语言处理研究计算机与人类语言之间的交互",
        "gold": ["自然语言处理", "研究", "计算机", "与", "人类语言", "之间", "的", "交互"],
    },
    {
        "text": "这款无线耳机音质很清晰",
        "gold": ["这款", "无线耳机", "音质", "很", "清晰"],
    },
    {
        "text": "人工智能技术正在加速进入教育医疗领域",
        "gold": ["人工智能", "技术", "正在", "加速", "进入", "教育", "医疗", "领域"],
    },
]

GOLD_POS = [
    ("自然语言处理", "n"),
    ("研究", "v"),
    ("计算机", "n"),
    ("与", "x"),
    ("人类语言", "n"),
    ("之间", "x"),
    ("交互", "v"),
    ("无线耳机", "n"),
    ("音质", "n"),
    ("清晰", "a"),
]


@dataclass(frozen=True)
class SegmentationResult:
    name: str
    tokens: list[str]
    elapsed_ms: float


def normalize_text(text: str, convert_traditional: bool, remove_symbols: bool) -> tuple[str, list[str]]:
    steps = []
    normalized = unicodedata.normalize("NFKC", text)
    if normalized != text:
        steps.append("全角字符已通过 NFKC 转为半角。")

    if convert_traditional and OpenCC is not None:
        converted = OpenCC("t2s").convert(normalized)
        if converted != normalized:
            steps.append("繁体字已转换为简体字。")
        normalized = converted
    elif convert_traditional:
        steps.append("当前环境未加载 OpenCC，已跳过繁简转换。")

    normalized = re.sub(r"\s+", " ", normalized).strip()
    if remove_symbols:
        cleaned = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9\s，。！？；：、,.!?;:，]", " ", normalized)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned != normalized:
            steps.append("特殊符号已过滤，保留中英文、数字和常见标点。")
        normalized = cleaned

    if not steps:
        steps.append("文本已检查，无需明显规范化。")
    return normalized, steps


def split_sentences_and_chars(text: str) -> list[str]:
    chunks = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+|[^\s]", text)
    return chunks


def build_dictionary(extra_terms: str) -> set[str]:
    dictionary = set(BASE_DICTIONARY)
    for term in re.split(r"[,，\n\s]+", extra_terms):
        if term.strip():
            dictionary.add(term.strip())
    for word in dictionary:
        jieba.add_word(word)
    return dictionary


def fmm_segment(text: str, dictionary: set[str], max_len: int = 8) -> list[str]:
    tokens = []
    chunks = split_sentences_and_chars(text)
    for chunk in chunks:
        if not re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            tokens.append(chunk)
            continue
        index = 0
        while index < len(chunk):
            window = min(max_len, len(chunk) - index)
            while window > 1 and chunk[index : index + window] not in dictionary:
                window -= 1
            tokens.append(chunk[index : index + window])
            index += window
    return tokens


def rmm_segment(text: str, dictionary: set[str], max_len: int = 8) -> list[str]:
    tokens = []
    chunks = split_sentences_and_chars(text)
    for chunk in chunks:
        if not re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            tokens.append(chunk)
            continue
        index = len(chunk)
        current = []
        while index > 0:
            window = min(max_len, index)
            while window > 1 and chunk[index - window : index] not in dictionary:
                window -= 1
            current.append(chunk[index - window : index])
            index -= window
        tokens.extend(reversed(current))
    return tokens


def bimm_segment(text: str, dictionary: set[str]) -> list[str]:
    forward = fmm_segment(text, dictionary)
    reverse = rmm_segment(text, dictionary)
    if len(forward) != len(reverse):
        return forward if len(forward) < len(reverse) else reverse
    f_singletons = sum(1 for token in forward if len(token) == 1)
    r_singletons = sum(1 for token in reverse if len(token) == 1)
    return forward if f_singletons <= r_singletons else reverse


def hybrid_segment(text: str, dictionary: set[str]) -> list[str]:
    rough_tokens = bimm_segment(text, dictionary)
    tokens = []
    for token in rough_tokens:
        if len(token) <= 2 or token in dictionary or not re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.append(token)
        else:
            tokens.extend(jieba.lcut(token, HMM=True))
    return tokens


def run_segmenter(name: str, text: str, dictionary: set[str]) -> SegmentationResult:
    start = time.perf_counter()
    if name == "FMM 正向最大匹配":
        tokens = fmm_segment(text, dictionary)
    elif name == "RMM 逆向最大匹配":
        tokens = rmm_segment(text, dictionary)
    elif name == "Bi-MM 双向最大匹配":
        tokens = bimm_segment(text, dictionary)
    elif name == "Jieba 精确模式":
        tokens = jieba.lcut(text, HMM=True)
    elif name == "Jieba 关闭 HMM":
        tokens = jieba.lcut(text, HMM=False)
    else:
        tokens = hybrid_segment(text, dictionary)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return SegmentationResult(name=name, tokens=clean_tokens(tokens), elapsed_ms=elapsed_ms)


def clean_tokens(tokens: Iterable[str]) -> list[str]:
    return [token.strip() for token in tokens if token and token.strip()]


def token_frequency(tokens: list[str]) -> pd.DataFrame:
    useful = [token for token in tokens if token not in STOPWORDS and re.search(r"[\u4e00-\u9fffA-Za-z0-9]", token)]
    counts = Counter(useful).most_common(10)
    return pd.DataFrame(counts, columns=["词语", "频次"])


def spans(tokens: list[str]) -> set[int]:
    boundaries = set()
    cursor = 0
    for token in tokens[:-1]:
        cursor += len(token)
        boundaries.add(cursor)
    return boundaries


def segmentation_f1(predicted: list[str], gold: list[str]) -> tuple[float, float, float]:
    pred_spans = spans(predicted)
    gold_spans = spans(gold)
    if not pred_spans and not gold_spans:
        return 1.0, 1.0, 1.0
    correct = len(pred_spans & gold_spans)
    precision = correct / len(pred_spans) if pred_spans else 0.0
    recall = correct / len(gold_spans) if gold_spans else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_segmenters(dictionary: set[str]) -> pd.DataFrame:
    rows = []
    algorithms = [
        "FMM 正向最大匹配",
        "RMM 逆向最大匹配",
        "Bi-MM 双向最大匹配",
        "Jieba 精确模式",
        "Jieba 关闭 HMM",
        "混合：Bi-MM + HMM",
    ]
    for algorithm in algorithms:
        precision_scores = []
        recall_scores = []
        f1_scores = []
        for sample in GOLD_SEGMENTATION:
            predicted = run_segmenter(algorithm, sample["text"], dictionary).tokens
            precision, recall, f1 = segmentation_f1(predicted, sample["gold"])
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        rows.append(
            {
                "分词算法": algorithm,
                "Precision": round(sum(precision_scores) / len(precision_scores), 3),
                "Recall": round(sum(recall_scores) / len(recall_scores), 3),
                "F1": round(sum(f1_scores) / len(f1_scores), 3),
            }
        )
    return pd.DataFrame(rows)


def rule_based_pos(tokens: list[str]) -> list[tuple[str, str]]:
    tags = []
    pronouns = {"这", "这款", "他", "她", "它", "我们", "你们"}
    for token in tokens:
        if token in POS_LEXICON:
            tag = POS_LEXICON[token]
        elif token.endswith(("好", "快", "短", "高", "低", "强")):
            tag = "a"
        elif token.endswith(("化", "入", "出", "取", "标注", "转换")):
            tag = "v"
        elif token in pronouns:
            tag = "r"
        elif token.isdigit():
            tag = "m"
        elif re.fullmatch(r"[A-Za-z0-9]+", token):
            tag = "eng"
        elif re.fullmatch(r"[，。！？；：、,.!?;:]", token):
            tag = "x"
        else:
            tag = "n"
        tags.append((token, tag))
    return tags


def jieba_pos(tokens: list[str], text: str) -> list[tuple[str, str]]:
    if not tokens:
        return []
    tagged_tokens = []
    for token in tokens:
        if token in POS_LEXICON:
            tagged_tokens.append((token, POS_LEXICON[token]))
            continue
        if re.fullmatch(r"[，。！？；：、,.!?;:]", token):
            tagged_tokens.append((token, "x"))
            continue
        candidates = [(word, flag) for word, flag in pseg.cut(token, HMM=True)]
        if not candidates:
            tagged_tokens.append((token, "x"))
            continue
        if len(candidates) == 1 and candidates[0][0] == token:
            tagged_tokens.append((token, normalize_pos_tag(candidates[0][1])))
            continue
        tagged_tokens.append((token, merge_pos_tags(flag for _, flag in candidates)))
    return tagged_tokens


def merge_pos_tags(tags: Iterable[str]) -> str:
    coarse_tags = [normalize_pos_tag(tag) for tag in tags]
    for priority_tag in ("n", "v", "a", "t", "m", "eng", "r"):
        if priority_tag in coarse_tags:
            return priority_tag
    return "x"


def normalize_pos_tag(tag: str) -> str:
    if tag.startswith("v"):
        return "v"
    if tag.startswith(("a", "d")):
        return "a"
    if tag.startswith(("r", "p", "c", "u")):
        return "r" if tag.startswith("r") else "x"
    if tag.startswith("m"):
        return "m"
    if tag.startswith("t"):
        return "t"
    if tag == "eng":
        return "eng"
    if tag.startswith("n"):
        return "n"
    return "x"


def evaluate_pos_taggers() -> pd.DataFrame:
    gold_tokens = [token for token, _ in GOLD_POS]
    gold_tags = dict(GOLD_POS)
    rows = []
    for name, tagged in [
        ("规则 baseline", rule_based_pos(gold_tokens)),
        ("Jieba HMM 统计模型", jieba_pos(gold_tokens, "".join(gold_tokens))),
    ]:
        comparable = [(word, tag) for word, tag in tagged if word in gold_tags]
        correct = sum(1 for word, tag in comparable if gold_tags[word] == tag)
        accuracy = correct / len(comparable) if comparable else 0.0
        rows.append({"词性标注算法": name, "Accuracy": round(accuracy, 3), "可比词数": len(comparable)})
    return pd.DataFrame(rows)


def highlight_pos(tagged_tokens: list[tuple[str, str]], selected_tags: list[str]) -> str:
    selected_codes = {code for code, (label, _) in TAG_META.items() if label in selected_tags}
    parts = []
    for token, tag in tagged_tokens:
        label, color = TAG_META.get(tag, TAG_META["x"])
        active = tag in selected_codes
        class_name = "pos-chip active" if active else "pos-chip muted"
        parts.append(
            f'<span class="{class_name}" style="--tag-color:{color}">'
            f"{html.escape(token)}<small>{label}</small></span>"
        )
    return '<div class="pos-board">' + "".join(parts) + "</div>"


def render_token_bubbles(tokens: list[str]) -> str:
    parts = []
    for index, token in enumerate(tokens, start=1):
        safe_token = html.escape(token)
        parts.append(f'<span class="token-chip"><small>{index:02d}</small>{safe_token}</span>')
    return '<div class="token-board">' + "".join(parts) + "</div>"


def render_algorithm_detail_tabs(info: dict[str, str]) -> None:
    tab_names = ["伪代码", "核心逻辑", "公式指标", "效果特点", "使用情景"]
    tabs = st.tabs(tab_names)
    for tab, name in zip(tabs, tab_names, strict=True):
        with tab:
            content = info[name]
            if name == "伪代码":
                st.code(content, language="text")
            else:
                st.markdown(f'<div class="algo-detail">{html.escape(content)}</div>', unsafe_allow_html=True)


def render_selected_algorithm_explainer(segmenter_name: str, pos_name: str) -> None:
    st.markdown("### 当前算法说明")
    seg_tab, pos_tab = st.tabs(["分词算法", "词性标注算法"])
    with seg_tab:
        st.markdown(f"#### {segmenter_name}")
        render_algorithm_detail_tabs(SEGMENTER_DETAILS[segmenter_name])
    with pos_tab:
        st.markdown(f"#### {pos_name}")
        render_algorithm_detail_tabs(POS_TAGGER_DETAILS[pos_name])


def render_algorithm_notes() -> None:
    st.markdown("### 算法说明与展示口径")
    st.markdown(
        """
        <div class="glass-panel">
            <p><strong>分词 baseline：</strong>FMM / RMM 属于词典贪心算法，速度快、可解释，但容易受词典覆盖率影响；Bi-MM 用正反两次匹配减少歧义；Jieba HMM 能识别部分未登录词，是课堂展示中很适合对比的统计模型 baseline。</p>
            <p><strong>词性 baseline：</strong>规则标注器模拟 Brill Tagger 的“先粗标、再按规则修正”思想；Jieba 词性标注作为 HMM 统计模型代表。CNN / BiLSTM / CRF / 感知器等更强模型适合放在说明区，后续如果有训练数据可以继续接入。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_lexical_app() -> None:
    st.markdown(
        """
        <section class="module-hero lexical-hero" style="--accent:#f59e0b">
            <p class="eyebrow">APP 01 · LEXICAL ANALYSIS</p>
            <h1>词法分析应用</h1>
            <p>从原始中文文本出发，依次完成文本规范化、中文分词、词频统计、词性标注和 baseline 效果对比。</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    sample_name = st.selectbox("选择默认示例背景", list(DEFAULT_TEXTS))
    text = st.text_area(
        "输入一段中文长文本",
        value=DEFAULT_TEXTS[sample_name],
        height=170,
        help="可以粘贴新闻、评论、论文摘要等不同风格文本。页面会自动完成后续词法分析工作流。",
    )

    control_cols = st.columns([1, 1, 1.2], gap="medium")
    with control_cols[0]:
        convert_traditional = st.toggle("繁体转简体", value=True)
        remove_symbols = st.toggle("去除特殊符号", value=True)
    with control_cols[1]:
        segmenter_name = st.selectbox(
            "分词算法",
            [
                "FMM 正向最大匹配",
                "RMM 逆向最大匹配",
                "Bi-MM 双向最大匹配",
                "Jieba 精确模式",
                "Jieba 关闭 HMM",
                "混合：Bi-MM + HMM",
            ],
            index=3,
        )
        pos_name = st.selectbox("词性标注算法", ["Jieba HMM 统计模型", "规则 baseline"], index=0)
    with control_cols[2]:
        extra_terms = st.text_area(
            "自定义词典词",
            value="",
            height=96,
            placeholder="例如：大语言模型、课程展示、命名实体",
            help="多个词可用空格、逗号或换行分隔，会同步用于 FMM/RMM/Bi-MM 和 Jieba。",
        )

    dictionary = build_dictionary(extra_terms)
    render_selected_algorithm_explainer(segmenter_name, pos_name)

    st.markdown("## 模块 1：文本规范化")
    normalized_text, steps = normalize_text(text, convert_traditional, remove_symbols)
    before, after = st.columns(2, gap="large")
    with before:
        st.markdown("#### 原始文本")
        st.markdown(f'<div class="text-panel">{html.escape(text)}</div>', unsafe_allow_html=True)
    with after:
        st.markdown("#### 规范化结果")
        st.markdown(f'<div class="text-panel normalized">{html.escape(normalized_text)}</div>', unsafe_allow_html=True)
    st.caption(" · ".join(steps))

    st.markdown("## 模块 2：中文分词与词频统计")
    result = run_segmenter(segmenter_name, normalized_text, dictionary)
    stat_cols = st.columns(4)
    stat_cols[0].metric("分词算法", result.name)
    stat_cols[1].metric("Token 数", len(result.tokens))
    stat_cols[2].metric("去重词数", len(set(result.tokens)))
    stat_cols[3].metric("耗时", f"{result.elapsed_ms:.2f} ms")

    st.markdown("#### 分词结果")
    st.markdown(render_token_bubbles(result.tokens), unsafe_allow_html=True)
    st.caption("每个泡泡代表一个分词 token，序号用于课堂展示时快速定位分词差异。")

    freq_df = token_frequency(result.tokens)
    chart_col, table_col = st.columns([1.35, 0.75], gap="large")
    with chart_col:
        if not freq_df.empty:
            fig = px.bar(
                freq_df.head(5),
                x="词语",
                y="频次",
                text="频次",
                color="频次",
                color_continuous_scale=["#fde68a", "#f97316"],
                title="Top 5 高频词",
            )
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("当前文本中没有可统计的有效词。")
    with table_col:
        st.dataframe(freq_df, use_container_width=True, hide_index=True)

    with st.expander("查看分词算法效能对比"):
        seg_eval = evaluate_segmenters(dictionary)
        st.dataframe(seg_eval, use_container_width=True, hide_index=True)
        st.caption("指标基于内置小型 gold set 的边界 Precision / Recall / F1，适合作课堂演示，不代表工业级评测。")

    st.markdown("## 模块 3：词性标注与同词性高亮")
    if pos_name == "规则 baseline":
        tagged_tokens = rule_based_pos(result.tokens)
    else:
        tagged_tokens = jieba_pos(result.tokens, normalized_text)

    tag_options = [label for label, _ in TAG_META.values()]
    selected_tags = st.multiselect("选择要高亮的词性", tag_options, default=["名词", "动词", "形容词"])
    st.markdown(highlight_pos(tagged_tokens, selected_tags), unsafe_allow_html=True)

    pos_table = pd.DataFrame(
        [
            {"词语": token, "词性": TAG_META.get(tag, TAG_META["x"])[0], "标记": tag}
            for token, tag in tagged_tokens
        ]
    )
    st.dataframe(pos_table, use_container_width=True, hide_index=True)

    with st.expander("查看词性标注 baseline 对比"):
        st.dataframe(evaluate_pos_taggers(), use_container_width=True, hide_index=True)
        st.caption("规则 baseline 用少量词典和后缀规则模拟 Brill 思路；Jieba HMM 统计模型作为可运行对照。")

    render_algorithm_notes()
