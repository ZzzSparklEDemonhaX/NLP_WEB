from __future__ import annotations

import html
import os
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
MODEL_CACHE_DIR = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--lxyuan--distilbert-base-multilingual-cased-sentiments-student"
)

DEFAULT_REVIEW = "这款耳机音质很棒，降噪效果也很好，物流速度非常快。"
DEFAULT_EXPLICIT = "这屏幕画质太垃圾了。"
DEFAULT_IMPLICIT = "在太阳底下根本看不清屏幕上的字。"

POSITIVE_WORDS = {"好", "棒", "喜欢", "优秀", "满意", "清晰", "顺滑", "推荐", "惊喜", "舒服", "快", "值得"}
NEGATIVE_WORDS = {"差", "垃圾", "失望", "糟糕", "卡", "慢", "坏", "贵", "模糊", "发烫", "没电", "看不清", "退货"}
IMPLICIT_NEGATIVE_PATTERNS = {"看不清", "半小时就没电", "一天要充三次", "放进口袋就发烫", "等了很久", "打不开"}

SAMPLE_REVIEWS = [
    "手机拍照很清晰，夜景效果比预期好，整体非常满意。",
    "物流很快，包装也完整，客服回复及时。",
    "电池续航太差了，玩游戏半小时就没电。",
    "屏幕在太阳底下根本看不清，体验很糟糕。",
    "价格有点贵，但性能还算稳定。",
    "外观漂亮，系统运行顺滑，值得推荐。",
    "用了两天就开始发烫，还经常卡顿。",
    "声音效果一般，没有特别惊喜。",
    "售后处理很慢，让人很失望。",
    "键盘手感舒服，办公效率提升明显。",
    "摄像头表现中规中矩，白天还可以。",
    "软件更新后闪退问题解决了，体验好多了。",
    "包装破损，配件也少了一根线。",
    "这个价位能有这样的质量，我觉得很划算。",
    "屏幕色彩偏冷，需要自己调节。",
]

LABEL_ZH = {
    "positive": "Positive 积极",
    "neutral": "Neutral 中性",
    "negative": "Negative 消极",
}

LABEL_COLOR = {
    "positive": "#22c55e",
    "neutral": "#f59e0b",
    "negative": "#ef4444",
}


@dataclass(frozen=True)
class SentimentResult:
    label: str
    score: float
    source: str


@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    """加载轻量级 Hugging Face 多语种情感模型，优先使用本地缓存。"""
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY"):
        os.environ.pop(key, None)
        os.environ.pop(key.lower(), None)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model_path = resolve_local_model_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.eval()
    return tokenizer, model


def resolve_local_model_path() -> str:
    """直接定位 Hugging Face snapshot 路径，避免 repo id 触发联网 model_info 检查。"""
    ref_path = MODEL_CACHE_DIR / "refs" / "main"
    if ref_path.exists():
        snapshot = MODEL_CACHE_DIR / "snapshots" / ref_path.read_text(encoding="utf-8").strip()
        if snapshot.exists():
            return str(snapshot)
    snapshots_dir = MODEL_CACHE_DIR / "snapshots"
    if snapshots_dir.exists():
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        if snapshots:
            return str(snapshots[0])
    return MODEL_NAME


def lexicon_sentiment(text: str) -> SentimentResult:
    """规则兜底：用中文情感词和隐式负面模式估计极性。"""
    pos = sum(1 for word in POSITIVE_WORDS if word in text)
    neg = sum(1 for word in NEGATIVE_WORDS if word in text)
    neg += sum(2 for pattern in IMPLICIT_NEGATIVE_PATTERNS if pattern in text)
    if pos > neg:
        score = min(0.55 + 0.12 * (pos - neg), 0.96)
        return SentimentResult("positive", score, "规则词典兜底")
    if neg > pos:
        score = min(0.55 + 0.12 * (neg - pos), 0.96)
        return SentimentResult("negative", score, "规则词典兜底")
    return SentimentResult("neutral", 0.58, "规则词典兜底")


def analyze_sentiment(text: str) -> SentimentResult:
    """运行情感分类；模型失败时退回规则词典。"""
    try:
        tokenizer, model = load_sentiment_model()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        index = int(torch.argmax(probs).item())
        label = model.config.id2label[index].lower()
        return SentimentResult(label, float(probs[index].detach().cpu()), "Hugging Face 模型")
    except Exception:
        return lexicon_sentiment(text)


def gauge_chart(result: SentimentResult):
    """用 Plotly 半圆仪表盘展示置信度。"""
    color = LABEL_COLOR.get(result.label, "#64748b")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=result.score * 100,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": LABEL_ZH.get(result.label, result.label), "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.4)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(239,68,68,0.14)"},
                    {"range": [50, 75], "color": "rgba(245,158,11,0.16)"},
                    {"range": [75, 100], "color": "rgba(34,197,94,0.16)"},
                ],
            },
        )
    )
    fig.update_layout(height=310, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def render_result_card(result: SentimentResult) -> str:
    color = LABEL_COLOR.get(result.label, "#64748b")
    return (
        f'<div class="sentiment-card" style="--sent-color:{color}">'
        f"<span>{html.escape(result.source)}</span>"
        f"<h3>{html.escape(LABEL_ZH.get(result.label, result.label))}</h3>"
        f"<p>Confidence: {result.score:.4f}</p>"
        "</div>"
    )


def batch_analyze(reviews: list[str]) -> pd.DataFrame:
    rows = []
    for review in reviews:
        result = analyze_sentiment(review)
        rows.append(
            {
                "评论": review,
                "情感标签": LABEL_ZH.get(result.label, result.label),
                "置信度": round(result.score, 4),
                "来源": result.source,
            }
        )
    return pd.DataFrame(rows)


def render_sentiment_intro() -> None:
    st.markdown(
        """
        <section class="module-hero sentiment-hero" style="--accent:#84cc16">
            <p class="eyebrow">APP 09 · SENTIMENT ANALYSIS</p>
            <h1>情感分析可视化应用</h1>
            <p>从单条评论的情感极性与置信度，到显式/隐式情感对比，再到批量舆情挖掘仪表盘。</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_single_tab() -> None:
    st.markdown("### 模块 1：基础情感分类与置信度量化")
    text = st.text_area("输入中文商品评论", value=DEFAULT_REVIEW, height=120)
    with st.spinner("正在分析情感极性..."):
        result = analyze_sentiment(text)
    left, right = st.columns([0.9, 1.2], gap="large")
    with left:
        st.markdown(render_result_card(result), unsafe_allow_html=True)
        st.caption("工程实践中，置信度能帮助判断是否需要人工复核。")
    with right:
        st.plotly_chart(gauge_chart(result), use_container_width=True)


def render_explicit_implicit_tab() -> None:
    st.markdown("### 模块 2：显式情感 vs 隐式情感识别")
    st.markdown(
        """
        <div class="sentiment-note">
            显式情感通常包含明显褒贬词，如“太棒了”“垃圾”；隐式情感没有直接情绪词，但通过客观事实暗示态度，如“半小时就没电了”。
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        explicit_text = st.text_area("显式情感评价", value=DEFAULT_EXPLICIT, height=100)
        explicit_result = analyze_sentiment(explicit_text)
        st.markdown(render_result_card(explicit_result), unsafe_allow_html=True)
    with col_b:
        implicit_text = st.text_area("隐式客观描述", value=DEFAULT_IMPLICIT, height=100)
        implicit_result = analyze_sentiment(implicit_text)
        st.markdown(render_result_card(implicit_result), unsafe_allow_html=True)
    st.caption("观察：小型模型对隐式负面更容易犹豫，规则兜底会额外关注“看不清、没电、发烫”等事实模式。")


def render_dashboard_tab() -> None:
    st.markdown("### 模块 3：舆情挖掘与可视化仪表盘")
    if st.button("生成测试舆情数据", use_container_width=True):
        st.session_state["opinion_reviews"] = random.sample(SAMPLE_REVIEWS, k=12)
    reviews = st.session_state.get("opinion_reviews", SAMPLE_REVIEWS[:12])
    custom = st.text_area("批量评论数据（每行一条）", value="\n".join(reviews), height=190)
    review_list = [line.strip() for line in custom.splitlines() if line.strip()]

    with st.spinner("正在批量分析舆情..."):
        df = batch_analyze(review_list)
    counts = df["情感标签"].value_counts().reset_index()
    counts.columns = ["情感标签", "数量"]

    chart_col, table_col = st.columns([0.95, 1.15], gap="large")
    with chart_col:
        fig = px.pie(
            counts,
            names="情感标签",
            values="数量",
            hole=0.45,
            color="情感标签",
            color_discrete_map={
                "Positive 积极": "#22c55e",
                "Neutral 中性": "#f59e0b",
                "Negative 消极": "#ef4444",
            },
        )
        fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with table_col:
        st.dataframe(df, use_container_width=True, hide_index=True)

    neg_count = int(counts.loc[counts["情感标签"] == "Negative 消极", "数量"].sum()) if "Negative 消极" in set(counts["情感标签"]) else 0
    if review_list and neg_count / len(review_list) >= 0.35:
        st.warning("负面评论比例偏高，建议关注产品质量、售后和续航等高频问题。")
    else:
        st.success("当前舆情整体较稳定，可继续观察中性评论中的潜在改进点。")


def render_sentiment_app() -> None:
    render_sentiment_intro()
    tab1, tab2, tab3 = st.tabs(["单句情感仪表盘", "显式 vs 隐式", "舆情挖掘大屏"])
    with tab1:
        try:
            render_single_tab()
        except Exception as exc:
            st.error(f"单句情感模块运行失败：{exc}")
    with tab2:
        try:
            render_explicit_implicit_tab()
        except Exception as exc:
            st.error(f"显隐式情感模块运行失败：{exc}")
    with tab3:
        try:
            render_dashboard_tab()
        except Exception as exc:
            st.error(f"舆情仪表盘模块运行失败：{exc}")
