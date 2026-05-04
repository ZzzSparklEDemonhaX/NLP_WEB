import streamlit as st


SAMPLE_TEXT = """自然语言处理让计算机能够理解、分析和生成 人类语言。
这个项目会把课程中的核心知识点变成可以交互演示的小应用。"""


def render_app_shell(title: str, subtitle: str, accent: str, demo_name: str) -> None:
    st.markdown(
        f"""
        <section class="module-hero" style="--accent:{accent}">
            <p class="eyebrow">NLP MINI APP</p>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown("### 输入区")
        user_text = st.text_area(
            "输入一段中文或英文文本",
            value=SAMPLE_TEXT,
            height=180,
            label_visibility="collapsed",
        )
        run = st.button(f"运行 {demo_name}", use_container_width=True)

    with right:
        st.markdown("### 展示目标")
        st.markdown(
            """
            <div class="glass-panel compact">
                <p><strong>这里会放置真实模型或算法输出。</strong></p>
                <p>后续每加入一个功能，我们会把输入、处理流程、结果解释和可视化统一放在这个页面里。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 当前占位演示")
    if run:
        tokens = [token for token in user_text.replace("\n", " ").split(" ") if token]
        st.success("演示运行成功。真实算法会在下一步接入。")
        st.write(
            {
                "输入字符数": len(user_text),
                "简单切分片段数": len(tokens),
                "示例片段": tokens[:8],
            }
        )
    else:
        st.info("点击运行按钮可以看到基础交互效果。")


def render_placeholder_word_lab() -> None:
    render_app_shell("词法分析实验室", "从分词、词频到关键词提取，展示文本最细粒度的结构。", "#f59e0b", "词法分析")


def render_placeholder_syntax_lab() -> None:
    render_app_shell("句法分析工坊", "把句子拆成可解释的结构，辅助理解语言中的组合关系。", "#06b6d4", "句法分析")


def render_placeholder_semantic_lab() -> None:
    render_app_shell("语义分析空间", "比较文本含义、计算相似度，并展示语义向量背后的直觉。", "#22c55e", "语义分析")


def render_placeholder_sentiment_lab() -> None:
    render_app_shell("情感分析仪表盘", "识别文本的积极、消极或中性倾向，并给出可展示的置信度。", "#ef4444", "情感分析")


def render_placeholder_topic_lab() -> None:
    render_app_shell("主题发现引擎", "从一组文档中发现主题，呈现关键词和文档聚类结果。", "#8b5cf6", "主题发现")


def render_placeholder_ie_lab() -> None:
    render_app_shell("信息抽取助手", "把非结构化文本转化为实体、关系和表格数据。", "#14b8a6", "信息抽取")


def render_placeholder_qa_lab() -> None:
    render_app_shell("问答系统沙盒", "基于上下文定位答案，让演示像一个可交互的阅读助手。", "#3b82f6", "问答系统")


def render_placeholder_summary_lab() -> None:
    render_app_shell("文本摘要控制台", "压缩长文本，保留关键信息，并支持摘要长度调节。", "#f97316", "文本摘要")


def render_placeholder_kg_lab() -> None:
    render_app_shell("知识图谱画布", "用节点和边展示文本中的实体关系，形成直观的知识网络。", "#84cc16", "知识图谱")
