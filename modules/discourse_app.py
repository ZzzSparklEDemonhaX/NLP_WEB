from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import spacy
import streamlit as st


NEURAL_EDU_RAW_URL = "https://raw.githubusercontent.com/PKU-TANGENT/NeuralEDUSeg/master/data/rst/TRAINING/wsj_0605.out"
NEURAL_EDU_GOLD_URL = "https://raw.githubusercontent.com/PKU-TANGENT/NeuralEDUSeg/master/data/rst/TRAINING/wsj_0605.out.edus"
LOCAL_EDU_RAW = Path(__file__).resolve().parents[1] / "data" / "neural_edu_sample.out"
LOCAL_EDU_GOLD = Path(__file__).resolve().parents[1] / "data" / "neural_edu_sample.out.edus"

FALLBACK_EDUS = [
    "The company said",
    "it expects sales to improve",
    "because new products are reaching customers",
    "although currency changes remain a risk.",
]

DEFAULT_PDTB_TEXT = (
    "Third-quarter sales in Europe were exceptionally strong, boosted by promotional programs and new products "
    "- although weaker foreign currencies reduced the company's earnings."
)

DEFAULT_COREF_TEXT = (
    "Barack Obama was born in Hawaii. He was elected president in 2008. "
    "Obama said his administration would focus on healthcare. Michelle Obama supported him, and she also worked on education."
)

CONNECTIVES = {
    "when": "TEMPORAL",
    "after": "TEMPORAL",
    "before": "TEMPORAL",
    "while": "TEMPORAL",
    "because": "CONTINGENCY",
    "since": "AMBIGUOUS",
    "so": "CONTINGENCY",
    "therefore": "CONTINGENCY",
    "but": "COMPARISON",
    "although": "COMPARISON",
    "however": "COMPARISON",
    "and": "EXPANSION",
    "or": "EXPANSION",
    "also": "EXPANSION",
}

COREF_COLORS = ["#fde68a", "#bfdbfe", "#fecdd3", "#bbf7d0", "#ddd6fe", "#fed7aa", "#a7f3d0"]


@dataclass(frozen=True)
class EduSample:
    text: str
    gold_edus: list[str]
    status: str


@dataclass(frozen=True)
class EduBoundary:
    token: str
    index: int
    reason: str


@st.cache_resource(show_spinner=False)
def load_spacy_en() -> tuple[spacy.language.Language, str]:
    """加载英文 spaCy 模型；如果模型不可用则退回 sentencizer。"""
    try:
        return spacy.load("en_core_web_sm"), "已加载 en_core_web_sm。"
    except OSError:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp, "未找到 en_core_web_sm，已退回基础句子切分器。"


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_neural_edu_sample(source_mode: str) -> EduSample:
    """从 NeuralEDUSeg GitHub 数据目录抓取一个 RST 样本；失败时使用内置样本。"""
    try:
        if source_mode == "本地缓存" and LOCAL_EDU_RAW.exists() and LOCAL_EDU_GOLD.exists():
            raw_text = normalize_space(LOCAL_EDU_RAW.read_text(encoding="utf-8"))
            gold_edus = parse_gold_edus(LOCAL_EDU_GOLD.read_text(encoding="utf-8"))
            return EduSample(raw_text or normalize_space(" ".join(gold_edus)), gold_edus[:18], "已加载本地缓存的 NeuralEDUSeg RST 样本。")

        session = requests.Session()
        session.trust_env = False
        raw_response = session.get(NEURAL_EDU_RAW_URL, timeout=12)
        gold_response = session.get(NEURAL_EDU_GOLD_URL, timeout=12)
        raw_response.raise_for_status()
        gold_response.raise_for_status()
        raw_text = normalize_space(raw_response.text)
        gold_edus = parse_gold_edus(gold_response.text)
        if not raw_text and gold_edus:
            raw_text = normalize_space(" ".join(gold_edus))
        if not gold_edus:
            raise ValueError("EDU 标注文件为空。")
        return EduSample(raw_text, gold_edus[:18], "已在线加载 NeuralEDUSeg RST 样本。")
    except Exception as exc:
        if LOCAL_EDU_RAW.exists() and LOCAL_EDU_GOLD.exists():
            raw_text = normalize_space(LOCAL_EDU_RAW.read_text(encoding="utf-8"))
            gold_edus = parse_gold_edus(LOCAL_EDU_GOLD.read_text(encoding="utf-8"))
            return EduSample(
                raw_text or normalize_space(" ".join(gold_edus)),
                gold_edus[:18],
                f"在线样本加载失败，已切换到本地缓存：{short_error(exc)}",
            )
        fallback_text = " ".join(FALLBACK_EDUS)
        return EduSample(fallback_text, FALLBACK_EDUS, f"在线样本加载失败，已使用内置 EDU 样本：{short_error(exc)}")


def short_error(exc: Exception) -> str:
    message = str(exc).replace("\n", " ").strip()
    if "Connection" in message or "WinError" in message or "timed out" in message:
        return "网络连接失败。"
    if len(message) > 120:
        return message[:120] + "..."
    return message or exc.__class__.__name__


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_gold_edus(text: str) -> list[str]:
    """解析 .out.edus：通常一行就是一个 EDU，去掉空行和残余标记。"""
    edus = []
    for line in text.splitlines():
        cleaned = normalize_space(re.sub(r"<[^>]+>", " ", line))
        if cleaned:
            edus.append(cleaned)
    return edus


def rule_edu_segment(text: str) -> tuple[list[str], set[str], list[EduBoundary]]:
    """规则基线：用标点、从属连词和依存标记近似寻找 EDU 边界。"""
    nlp, _ = load_spacy_en()
    doc = nlp(text)
    boundary_indices = set()
    boundary_words = set()
    boundary_reasons: dict[int, list[str]] = {}
    connective_words = {"because", "although", "while", "when", "if", "but", "however", "since"}

    def add_boundary(index: int, token_text: str, reason: str) -> None:
        if index < 0 or index >= len(doc):
            return
        boundary_indices.add(index)
        boundary_words.add(token_text)
        boundary_reasons.setdefault(index, []).append(reason)

    for token in doc:
        lower = token.text.lower()
        if token.text in {".", ";", ":", "!", "?"}:
            add_boundary(token.i, token.text, "句末/强标点通常结束一个 EDU")
        if lower in connective_words or token.pos_ == "SCONJ" or token.dep_ in {"mark", "advcl", "relcl"}:
            add_boundary(max(token.i - 1, 0), token.text, f"连接词/从句线索 `{token.text}` 触发边界")

    edus = []
    start = 0
    for index, token in enumerate(doc):
        if index in boundary_indices:
            span = doc[start : index + 1].text.strip()
            if span:
                edus.append(span)
            start = index + 1
    if start < len(doc):
        span = doc[start:].text.strip()
        if span:
            edus.append(span)
    reasons = [
        EduBoundary(doc[index].text, index, "；".join(dict.fromkeys(boundary_reasons.get(index, ["规则边界"]))))
        for index in sorted(boundary_indices)
    ]
    return edus or [text], boundary_words, reasons


def edu_end_token(edu: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9']+|[^\w\s]", edu)
    content_tokens = [token for token in tokens if re.search(r"[A-Za-z0-9]", token)]
    return content_tokens[-1] if content_tokens else (tokens[-1] if tokens else "")


def render_edu_cards(edus: list[str], boundary_words: set[str] | None = None, highlight_last: bool = True) -> str:
    boundary_words = boundary_words or set()
    cards = []
    for index, edu in enumerate(edus, start=1):
        rendered = highlight_boundary_words(edu, boundary_words, edu_end_token(edu) if highlight_last else "")
        cards.append(f'<article class="edu-card"><span>EDU {index:02d}</span><p>{rendered}</p></article>')
    return '<div class="edu-list">' + "".join(cards) + "</div>"


def highlight_boundary_words(text: str, boundary_words: set[str], end_word: str = "") -> str:
    escaped = html.escape(text)
    for word in sorted(boundary_words, key=len, reverse=True):
        if not word or not re.search(r"\w", word):
            continue
        escaped = re.sub(
            rf"\b{re.escape(html.escape(word))}\b",
            lambda match: f'<mark class="boundary-token">{match.group(0)}</mark>',
            escaped,
            flags=re.IGNORECASE,
        )
    if end_word and re.search(r"\w", end_word):
        escaped = re.sub(
            rf"\b{re.escape(html.escape(end_word))}\b(?![^<]*>)",
            lambda match: f'<mark class="edu-end-token">{match.group(0)}</mark>',
            escaped,
            count=1,
            flags=re.IGNORECASE,
        )
    return escaped


def edu_reason_table(boundaries: list[EduBoundary]) -> pd.DataFrame:
    if not boundaries:
        return pd.DataFrame([{"边界词": "-", "Token 位置": "-", "切分原因": "没有触发显式边界，整段作为一个 EDU。"}])
    return pd.DataFrame(
        [
            {"边界词": item.token, "Token 位置": item.index, "切分原因": item.reason}
            for item in boundaries
        ]
    )


def compare_edu_segmentations(rule_edus: list[str], gold_edus: list[str]) -> pd.DataFrame:
    rows = []
    max_len = max(len(rule_edus), len(gold_edus))
    for index in range(max_len):
        rule_edu = rule_edus[index] if index < len(rule_edus) else ""
        gold_edu = gold_edus[index] if index < len(gold_edus) else ""
        rule_end = edu_end_token(rule_edu)
        gold_end = edu_end_token(gold_edu)
        rows.append(
            {
                "序号": index + 1,
                "规则基线 EDU": rule_edu or "-",
                "规则末词": rule_end or "-",
                "真实标注 EDU": gold_edu or "-",
                "真实末词": gold_end or "-",
                "差异": "末词一致" if rule_end and rule_end.lower() == gold_end.lower() else "边界不同",
            }
        )
    return pd.DataFrame(rows)


def is_sample_text(text: str, sample: EduSample) -> bool:
    return normalize_space(text) == normalize_space(sample.text)


def detect_connective(text: str) -> tuple[str | None, str | None, tuple[int, int] | None]:
    """扫描显式连接词，并对 since 做简易消歧。"""
    for connective in sorted(CONNECTIVES, key=len, reverse=True):
        match = re.search(rf"\b{re.escape(connective)}\b", text, flags=re.IGNORECASE)
        if not match:
            continue
        category = CONNECTIVES[connective]
        if connective == "since":
            following = text[match.end() : match.end() + 40].lower()
            if re.search(r"\b(then|yesterday|last|year|month|week|\d{4})\b", following):
                category = "TEMPORAL"
            else:
                category = "CONTINGENCY"
        return connective, category, match.span()
    return None, None, None


def render_pdtb_highlight(text: str, span: tuple[int, int] | None, category: str | None) -> str:
    if span is None or category is None:
        return f'<div class="pdtb-highlight">{html.escape(text)}</div>'
    start, end = span
    return (
        '<div class="pdtb-highlight">'
        + html.escape(text[:start])
        + f'<mark class="pdtb-connective">{html.escape(text[start:end])} [{category}]</mark>'
        + html.escape(text[end:])
        + "</div>"
    )


def split_arguments(text: str, span: tuple[int, int] | None) -> tuple[str, str]:
    if span is None:
        return text, ""
    start, end = span
    arg1 = text[:start].strip(" -,\n")
    arg2 = text[end:].strip(" -,\n")
    return arg1, arg2


@st.cache_resource(show_spinner=False)
def load_coref_model(enable_fastcoref: bool):
    """加载 fastcoref；失败时返回 None，页面使用规则兜底。"""
    if not enable_fastcoref:
        return None, "当前使用规则兜底。勾选“启用 fastcoref 神经模型”可运行真实模型。"
    try:
        # 当前实验环境可能残留不可用代理，fastcoref 会因此在 Hugging Face HEAD 检查时长时间重试。
        # 清理代理后优先使用本地缓存模型，避免课堂展示时卡住。
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY"):
            os.environ.pop(key, None)
            os.environ.pop(key.lower(), None)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from fastcoref import FCoref

        return FCoref(device="cpu"), "已加载 fastcoref FCoref 模型。"
    except Exception as exc:
        return None, f"fastcoref 模型加载失败，已使用规则兜底：{short_error(exc)}"


def run_coreference(text: str, enable_fastcoref: bool) -> tuple[list[list[tuple[int, int]]], list[list[str]], str]:
    model, status = load_coref_model(enable_fastcoref)
    if model is not None:
        try:
            prediction = model.predict([text])[0]
            spans = prediction.get_clusters(as_strings=False)
            strings = prediction.get_clusters(as_strings=True)
            return spans, strings, status
        except Exception as exc:
            status = f"fastcoref 预测失败，已使用规则兜底：{short_error(exc)}"
    spans, strings = fallback_coref(text)
    return spans, strings, status


def fallback_coref(text: str) -> tuple[list[list[tuple[int, int]]], list[list[str]]]:
    """非常轻量的指代兜底：把最近的人名和 he/she/his/her 聚成一类。"""
    mentions = []
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b|\b(he|she|his|her|him|they|them|it)\b", text):
        mentions.append((match.group(0), match.start(), match.end()))
    if not mentions:
        return [], []
    clusters = [[(start, end) for _, start, end in mentions]]
    strings = [[mention for mention, _, _ in mentions]]
    return clusters, strings


def render_coref_html(text: str, clusters: list[list[tuple[int, int]]]) -> str:
    """把同一簇 mention 用同一种颜色高亮。"""
    spans = []
    for cluster_index, cluster in enumerate(clusters):
        for start, end in cluster:
            spans.append((start, end, cluster_index))
    spans.sort(key=lambda item: item[0])

    rendered = []
    cursor = 0
    for start, end, cluster_index in spans:
        if start < cursor:
            continue
        rendered.append(html.escape(text[cursor:start]))
        color = COREF_COLORS[cluster_index % len(COREF_COLORS)]
        rendered.append(
            f'<mark class="coref-mention" style="--coref-color:{color}">'
            f"{html.escape(text[start:end])}<small>C{cluster_index + 1}</small></mark>"
        )
        cursor = end
    rendered.append(html.escape(text[cursor:]))
    return '<div class="coref-text">' + "".join(rendered) + "</div>"


def render_discourse_intro() -> None:
    st.markdown(
        """
        <section class="module-hero discourse-hero" style="--accent:#8b5cf6">
            <p class="eyebrow">APP 05 · DISCOURSE ANALYSIS</p>
            <h1>篇章分析综合平台</h1>
            <p>从 EDU 话语单元、PDTB 显式连接词到指代消解聚类，观察篇章衔接与连贯性的程序化分析。</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_edu_tab() -> None:
    st.markdown("### 模块 1：话语分割 EDU 切分")
    source_mode = st.radio(
        "NeuralEDUSeg 样本来源",
        ["在线联网", "本地缓存"],
        horizontal=True,
        help="在线联网会从 GitHub 原始数据目录抓取样本；本地缓存会读取项目 data 目录下已保存的样本。",
    )
    sample = fetch_neural_edu_sample(source_mode)
    st.caption(sample.status)
    st.markdown(
        """
        <div class="edu-explain">
            <b>规则基线切分</b>会对输入框中的任意文本实时分析；<b>NeuralEDUSeg 数据真实标注</b>来自公开 RST 样本文件，
            只能作为该样本文本的真实边界参照，不能自动给任意新输入生成人工标注。
        </div>
        """,
        unsafe_allow_html=True,
    )
    text = st.text_area("待切分篇章文本", value=sample.text, height=145)
    baseline_edus, boundary_words, boundary_reasons = rule_edu_segment(text)
    gold_matches_input = is_sample_text(text, sample)
    gold_edus = sample.gold_edus if gold_matches_input else []
    boundary_examples = [word for word in sorted(boundary_words) if re.search(r"\w", word)]
    boundary_example = boundary_examples[0] if boundary_examples else "boundary"
    end_examples = [token for token in [edu_end_token(item) for item in baseline_edus] if token]
    end_example = end_examples[0] if end_examples else "end"

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### 规则基线切分")
        st.markdown(render_edu_cards(baseline_edus, boundary_words, highlight_last=True), unsafe_allow_html=True)
    with right:
        st.markdown("#### NeuralEDUSeg 数据真实标注")
        if gold_matches_input:
            st.markdown(render_edu_cards(sample.gold_edus, highlight_last=True), unsafe_allow_html=True)
        else:
            st.warning("当前输入文本不是 NeuralEDUSeg 样本文本，因此没有可一一对应的真实 EDU 标注。右侧保留样本标注说明，不把它误当成当前输入的结果。")
            with st.expander("查看 NeuralEDUSeg 样本文本的真实标注"):
                st.markdown(render_edu_cards(sample.gold_edus, highlight_last=True), unsafe_allow_html=True)

    st.markdown("#### 规则基线切分逻辑与原因")
    st.dataframe(edu_reason_table(boundary_reasons), use_container_width=True, hide_index=True)
    st.markdown(
        """
        <div class="edu-legend">
            <span><mark class="boundary-token">because</mark> 边界触发词 / 连接词线索</span>
            <span><mark class="edu-end-token">said</mark> EDU 末词，即当前 EDU 的结束位置</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 两种方法切分差异")
    if gold_matches_input:
        diff_df = compare_edu_segmentations(baseline_edus, gold_edus)
        st.dataframe(diff_df, use_container_width=True, hide_index=True)
        same_count = int((diff_df["差异"] == "末词一致").sum())
        st.caption(f"规则基线 EDU 数：{len(baseline_edus)}；真实标注 EDU 数：{len(gold_edus)}；末词一致边界数：{same_count}。")
    else:
        st.info("差异比对需要当前输入与 NeuralEDUSeg 样本文本一致。若要观察真实标注对比，请恢复/保留默认样本文本；若输入自定义文本，页面只展示实时规则基线结果。")

    st.info("观察：规则基线更依赖标点、连接词和从句依存标签，因此可解释但较粗；NeuralEDUSeg 真实标注来自序列标注数据，边界通常更细，适合用来观察规则方法漏切或误切的位置。")


def render_pdtb_tab() -> None:
    st.markdown("### 模块 2：浅层篇章分析与显式关系提取")
    text = st.text_area("输入包含显式连接词的句子", value=DEFAULT_PDTB_TEXT, height=120)
    connective, category, span = detect_connective(text)
    st.markdown(render_pdtb_highlight(text, span, category), unsafe_allow_html=True)

    if connective is None:
        st.warning("未检测到内置连接词列表中的显式连接词。")
        return

    arg1, arg2 = split_arguments(text, span)
    st.metric("显式连接词", f"{connective} [{category}]")
    cols = st.columns(2, gap="large")
    with cols[0]:
        st.markdown(f'<div class="arg-card arg1"><span>Arg1</span><p>{html.escape(arg1)}</p></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'<div class="arg-card arg2"><span>Arg2</span><p>{html.escape(arg2)}</p></div>', unsafe_allow_html=True)

    st.caption("since 消歧提示：since + 时间表达常被判为 TEMPORAL；否则更可能是 CONTINGENCY。可尝试 Since 2010, ... 与 Since it rained, ...。")


def render_coref_tab() -> None:
    st.markdown("### 模块 3：指代消解 Coreference Resolution")
    text = st.text_area("输入包含代词和多次指称的英文段落", value=DEFAULT_COREF_TEXT, height=150)
    enable_fastcoref = st.checkbox(
        "启用 fastcoref 神经模型（首次加载可能较慢，需要模型缓存或网络）",
        value=True,
    )
    with st.spinner("正在运行指代消解模型..."):
        spans, strings, status = run_coreference(text, enable_fastcoref)
    st.caption(status)
    st.markdown(render_coref_html(text, spans), unsafe_allow_html=True)

    rows = [{"Cluster": f"Cluster {index + 1}", "Mentions": ", ".join(cluster)} for index, cluster in enumerate(strings)]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("没有识别到指代簇。")


def render_edu_tab() -> None:
    st.markdown("### 模块 1：话语分割 EDU 切分")
    source_mode = st.radio(
        "NeuralEDUSeg 样本来源",
        ["在线联网", "本地缓存"],
        horizontal=True,
        help="在线联网会从 GitHub 原始数据目录抓取样本；本地缓存会读取项目 data 目录下已保存的样本。",
    )
    sample = fetch_neural_edu_sample(source_mode)
    st.caption(sample.status)
    st.markdown(
        """
        <div class="edu-explain">
            <b>规则基线切分</b>会对输入框中的任意文本实时分析；<b>NeuralEDUSeg 数据真实标注</b>来自公开 RST 样本文件，
            只能作为该样本文本的真实边界参照，不能自动给任意新输入生成人工标注。
        </div>
        """,
        unsafe_allow_html=True,
    )
    text = st.text_area("待切分篇章文本", value=sample.text, height=145)
    baseline_edus, boundary_words, boundary_reasons = rule_edu_segment(text)
    gold_matches_input = is_sample_text(text, sample)
    gold_edus = sample.gold_edus if gold_matches_input else []
    boundary_examples = [word for word in sorted(boundary_words) if re.search(r"\w", word)]
    boundary_example = boundary_examples[0] if boundary_examples else "boundary"
    end_examples = [token for token in [edu_end_token(item) for item in baseline_edus] if token]
    end_example = end_examples[0] if end_examples else "end"

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### 规则基线切分")
        st.markdown(render_edu_cards(baseline_edus, boundary_words, highlight_last=True), unsafe_allow_html=True)
    with right:
        st.markdown("#### NeuralEDUSeg 数据真实标注")
        if gold_matches_input:
            st.markdown(render_edu_cards(gold_edus, highlight_last=True), unsafe_allow_html=True)
        else:
            st.warning("当前输入文本不是 NeuralEDUSeg 样本文本，因此没有可一一对应的真实 EDU 标注。")
            with st.expander("查看 NeuralEDUSeg 样本文本的真实标注"):
                st.markdown(render_edu_cards(sample.gold_edus, highlight_last=True), unsafe_allow_html=True)

    st.markdown("#### 规则基线切分逻辑与原因")
    st.dataframe(edu_reason_table(boundary_reasons), use_container_width=True, hide_index=True)
    st.markdown(
        f"""
        <div class="edu-legend">
            <span><mark class="boundary-token">{html.escape(boundary_example)}</mark> 当前文本里真正触发规则切分的边界词 / 连接词</span>
            <span><mark class="edu-end-token">{html.escape(end_example)}</mark> 当前规则切分结果中的 EDU 末词</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 两种方法切分差异")
    if gold_matches_input:
        diff_df = compare_edu_segmentations(baseline_edus, gold_edus)
        st.dataframe(diff_df, use_container_width=True, hide_index=True)
        same_count = int((diff_df["差异"] == "末词一致").sum())
        st.caption(f"规则基线 EDU 数：{len(baseline_edus)}；真实标注 EDU 数：{len(gold_edus)}；末词一致边界数：{same_count}。")
    else:
        st.info("差异比对需要当前输入与 NeuralEDUSeg 样本文本一致。输入自定义文本时，页面只展示实时规则基线结果。")

    st.info("观察：规则基线更依赖标点、连接词和从句依存标签，因此可解释但较粗；NeuralEDUSeg 真实标注更适合用来观察规则方法漏切或误切的位置。")


def render_discourse_app() -> None:
    render_discourse_intro()
    tab_edu, tab_pdtb, tab_coref = st.tabs(["EDU 话语分割", "PDTB 显式关系", "指代消解"])
    with tab_edu:
        try:
            render_edu_tab()
        except Exception as exc:
            st.error(f"EDU 模块运行失败：{short_error(exc)}")
    with tab_pdtb:
        try:
            render_pdtb_tab()
        except Exception as exc:
            st.error(f"PDTB 模块运行失败：{short_error(exc)}")
    with tab_coref:
        try:
            render_coref_tab()
        except Exception as exc:
            st.error(f"指代消解模块运行失败：{short_error(exc)}")
