from collections.abc import Sequence
from textwrap import dedent

import streamlit as st

from modules.registry import NlpApp


def render_sidebar(apps: Sequence[NlpApp]) -> NlpApp | None:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <span>NLP</span>
            <strong>Lab Studio</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    choice = st.sidebar.radio(
        "导航",
        options=["项目总览", *[app.title for app in apps]],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")

    if choice == "项目总览":
        return None
    return next(app for app in apps if app.title == choice)


def hero_section() -> None:
    st.markdown(
        dedent(
            """
            <section class="hero">
                <div>
                    <p class="eyebrow">NLP COURSE SYSTEM</p>
                    <h1>自然语言处理九模块<br/>知识架构总览</h1>
                    <p class="hero-copy">
                        这个项目不是九个彼此割裂的小页面，而是沿着自然语言处理的知识链路逐层展开：
                        从词法、句法，到语义、篇章，再到语言模型、信息抽取、机器翻译与情感分析，
                        最后形成一个适合课程展示的完整 Web 实验平台。
                    </p>
                </div>
                <div class="hero-orb">
                    <span>9</span>
                    <small>modules</small>
                </div>
            </section>
            """
        ),
        unsafe_allow_html=True,
    )


def metric_strip() -> None:
    cols = st.columns(3)
    metrics = [
        ("结构解析层", "词法分析与句法分析，回答“文本怎样被切分和组织”。"),
        ("语义理解层", "语义空间、深层语义、篇章分析，回答“文本真正表达了什么”。"),
        ("任务应用层", "语言模型、信息抽取、机器翻译、情感分析，回答“模型怎样生成与落地”。"),
    ]
    for col, (value, label) in zip(cols, metrics, strict=True):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <strong>{value}</strong>
                    <span>{label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_app_card(app: NlpApp) -> str:
    stage_tag = app.key.replace("_lab", "").replace("_", " ").upper()
    return dedent(
        f"""
        <article class="app-card" style="--accent:{app.accent}">
            <div class="app-card-topline">
                <span class="app-card-icon">{app.icon}</span>
                <small class="app-card-tag">{stage_tag}</small>
            </div>
            <h3>{app.title}</h3>
            <p>{app.subtitle}</p>
        </article>
        """
    ).strip()


def app_card_grid(apps: Sequence[NlpApp]) -> None:
    groups = [
        (
            "基础结构层",
            "文本先经过规范化、分词、词性标注，再进入句法结构分析，形成后续语义计算的输入基础。",
            [apps[0], apps[1]],
            "#f59e0b",
        ),
        (
            "语义理解层",
            "从词向量、词义消歧、语义角色到篇章衔接，逐步逼近句子与段落背后的真实含义。",
            [apps[2], apps[3], apps[4]],
            "#22c55e",
        ),
        (
            "生成建模层",
            "语言模型与机器翻译把表示学习推进到生成任务，体现从统计方法到预训练模型的演化。",
            [apps[5], apps[7]],
            "#14b8a6",
        ),
        (
            "知识应用层",
            "信息抽取和情感分析把自然语言理解结果转成结构化知识与宏观业务洞察。",
            [apps[6], apps[8]],
            "#3b82f6",
        ),
    ]

    st.markdown("## NLP 知识链路")
    for title, desc, group_apps, accent in groups:
        cards = "".join(render_app_card(app) for app in group_apps)
        html_block = dedent(
            f"""
            <section class="overview-stage" style="--accent:{accent}">
                <header>
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </header>
                <div class="overview-stage-grid">{cards}</div>
            </section>
            """
        ).strip()
        st.markdown(html_block, unsafe_allow_html=True)


def render_footer() -> None:
    return
