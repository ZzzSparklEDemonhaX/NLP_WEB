from __future__ import annotations

import html
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import spacy
import streamlit as st
from nltk import CFG, ChartParser, Tree
from spacy import displacy


# 这些示例都来自课堂常见的结构歧义讨论，方便展示时一键切换。
DEFAULT_SENTENCES = {
    "望远镜歧义": "The boy saw the man with the telescope.",
    "果蝇与香蕉": "Fruit flies like a banana.",
    "中文：南京市长江大桥": "南京市长江大桥",
    "中文：猎人与狗": "咬死了猎人的狗",
}

# 依存句法里常见的核心论元。spaCy 新模型里直接宾语有时叫 obj，旧标签常叫 dobj。
CORE_DEPS = {"nsubj", "nsubjpass", "dobj", "obj", "pobj", "ROOT"}

# benepar 只适合英文成分分析；中文部分先展示依存分析和歧义观察。
BENEPAR_MODEL = "benepar_en3"


@dataclass(frozen=True)
class ParserResource:
    nlp: spacy.language.Language
    model_name: str
    status: str
    has_dependency_parser: bool


@dataclass(frozen=True)
class ChineseSyntaxFallback:
    svg: str
    features: pd.DataFrame
    core_args: pd.DataFrame
    observations: list[dict[str, str]]
    status: str


def contains_chinese(text: str) -> bool:
    """判断输入是否包含中文字符，用于自动选择 English / Chinese spaCy 模型。"""
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def run_module_command(args: list[str]) -> tuple[bool, str]:
    """在当前 Python 环境中执行模型下载命令，并把错误转成页面可读的信息。"""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
        )
    except Exception as exc:  # pragma: no cover - 主要用于本地网络/权限异常兜底。
        return False, str(exc)
    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
    return completed.returncode == 0, output[-1200:]


@st.cache_resource(show_spinner=False)
def load_spacy_model(language: str) -> ParserResource:
    """加载 spaCy 模型；若缺失则尝试下载，下载失败时退回 blank tokenizer。"""
    model_name = "zh_core_web_sm" if language == "zh" else "en_core_web_sm"
    try:
        nlp = spacy.load(model_name)
        return ParserResource(nlp, model_name, "已加载本地模型。", "parser" in nlp.pipe_names)
    except OSError:
        ok, output = run_module_command(["spacy", "download", model_name])
        if ok:
            try:
                nlp = spacy.load(model_name)
                return ParserResource(nlp, model_name, "模型缺失，已自动下载并加载。", "parser" in nlp.pipe_names)
            except OSError as exc:
                output = f"{output}\n加载失败：{exc}"

        # 中文模型经常没装或下载失败。退回 blank 后仍可做分词展示，但不能给出真实依存关系。
        nlp = spacy.blank(language)
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return ParserResource(
            nlp,
            f"blank_{language}",
            f"未能加载 {model_name}，已退回基础分词器。依存图会降级展示。错误摘要：{short_message(output)}",
            False,
        )


@st.cache_resource(show_spinner=False)
def load_benepar_pipeline() -> tuple[spacy.language.Language | None, str]:
    """加载 benepar 英文成分句法管线；失败时页面会自动使用 NLTK CFG 兜底。"""
    # benepar 的部分依赖在新版 protobuf 下可能报错；设置该环境变量可以让它走兼容实现。
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    try:
        import benepar
    except ImportError:
        return None, "No module named 'benepar'。请在当前虚拟环境中安装 requirements.txt 后重试。"

    # 第一次使用时尝试下载 benepar_en3；若网络受限，后续会自动走 CFG fallback。
    try:
        import nltk.data

        nltk.data.find(f"models/{BENEPAR_MODEL}")
        download_note = "benepar 模型已准备。"
    except LookupError:
        try:
            benepar.download(BENEPAR_MODEL)
            download_note = "benepar 模型缺失，已尝试自动下载。"
        except Exception as exc:
            download_note = f"benepar 模型下载失败，将使用 NLTK CFG 兜底。错误摘要：{short_error(exc)}"

    try:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        if "benepar" not in nlp.pipe_names:
            nlp.add_pipe("benepar", config={"model": BENEPAR_MODEL})
        return nlp, download_note
    except Exception as exc:
        return None, f"{download_note}\n加载 benepar 管线失败，将使用 NLTK CFG 兜底：{short_error(exc)}"


def short_error(exc: Exception) -> str:
    """把很长的安装/模型错误压缩成适合课堂页面展示的一句话。"""
    message = str(exc).replace("\n", " ").strip()
    return short_message(message) or exc.__class__.__name__


def short_message(message: str) -> str:
    """压缩命令行输出，避免把完整 traceback 展示到课堂页面。"""
    message = message.replace("\n", " ").strip()
    if "Descriptors cannot be created directly" in message:
        return "protobuf 兼容性问题。"
    if "No matching distribution" in message:
        return "当前 Python 环境未找到可安装的模型包。"
    if "Read timed out" in message or "Connection" in message or "WinError" in message:
        return "模型下载网络连接失败。"
    if "No module named" in message:
        return message
    if len(message) > 140:
        return message[:140] + "..."
    return message


def parse_dependency(text: str, language_choice: str) -> tuple[spacy.tokens.Doc, ParserResource]:
    """根据用户选择或文本内容运行 spaCy 依存分析。"""
    language = "zh" if language_choice == "中文" or (language_choice == "自动识别" and contains_chinese(text)) else "en"
    resource = load_spacy_model(language)
    return resource.nlp(text), resource


def dependency_svg(doc: spacy.tokens.Doc) -> str | None:
    """把 spaCy Doc 渲染为 displaCy SVG。没有依存标签时返回 None。"""
    if not any(token.dep_ for token in doc):
        return None
    options = {
        "compact": True,
        "bg": "transparent",
        "color": "#18212f",
        "font": "Source Sans Pro",
        "distance": 92,
        "arrow_stroke": 2,
    }
    return displacy.render(doc, style="dep", options=options, jupyter=False)


def manual_dependency_svg(words: list[dict[str, str]], arcs: list[dict[str, str | int]]) -> str:
    """使用 displaCy 的 manual 模式绘制规则依存图，作为中文模型缺失时的兜底。"""
    options = {
        "compact": True,
        "bg": "transparent",
        "color": "#18212f",
        "font": "Source Sans Pro",
        "distance": 108,
        "arrow_stroke": 2,
    }
    return displacy.render({"words": words, "arcs": arcs}, style="dep", manual=True, options=options, jupyter=False)


def chinese_syntax_fallback(text: str) -> ChineseSyntaxFallback | None:
    """为课堂指定中文歧义句提供规则句法分析兜底，保证没有中文模型时也能展示。"""
    compact_text = "".join(char for char in text if "\u4e00" <= char <= "\u9fff")

    if compact_text == "南京市长江大桥":
        words = [
            {"text": "南京市", "tag": "PROPN"},
            {"text": "长江大桥", "tag": "NOUN"},
        ]
        arcs = [{"start": 0, "end": 1, "label": "nmod:loc", "dir": "left"}]
        features = pd.DataFrame(
            [
                {"序号": 1, "词": "南京市", "词形还原": "南京市", "词性 POS": "PROPN", "细粒度标签": "地名", "依存关系": "nmod:loc", "支配词 Head": "长江大桥"},
                {"序号": 2, "词": "长江大桥", "词形还原": "长江大桥", "词性 POS": "NOUN", "细粒度标签": "名词", "依存关系": "ROOT", "支配词 Head": "ROOT"},
            ]
        )
        core_args = pd.DataFrame(
            [{"词": "长江大桥", "依存关系": "ROOT", "说明": "中心名词 / 根节点", "支配词": "ROOT", "词性": "NOUN"}]
        )
        observations = [
            {
                "侦探任务": "中文歧义：南京市长江大桥",
                "观察": "规则兜底把它解释为“南京市的长江大桥”，即地名修饰桥名；它也可被错误切成“南京市长 / 江大桥”。",
            }
        ]
        return ChineseSyntaxFallback(manual_dependency_svg(words, arcs), features, core_args, observations, "中文规则兜底：地名修饰中心名词。")

    if compact_text == "咬死了猎人的狗":
        words = [
            {"text": "咬死了", "tag": "VERB"},
            {"text": "猎人", "tag": "NOUN"},
            {"text": "的", "tag": "PART"},
            {"text": "狗", "tag": "NOUN"},
        ]
        arcs = [
            {"start": 0, "end": 3, "label": "nsubj", "dir": "right"},
            {"start": 0, "end": 1, "label": "obj", "dir": "right"},
            {"start": 1, "end": 2, "label": "mark", "dir": "right"},
        ]
        features = pd.DataFrame(
            [
                {"序号": 1, "词": "咬死了", "词形还原": "咬死", "词性 POS": "VERB", "细粒度标签": "动词", "依存关系": "ROOT", "支配词 Head": "ROOT"},
                {"序号": 2, "词": "猎人", "词形还原": "猎人", "词性 POS": "NOUN", "细粒度标签": "名词", "依存关系": "obj", "支配词 Head": "咬死了"},
                {"序号": 3, "词": "的", "词形还原": "的", "词性 POS": "PART", "细粒度标签": "结构助词", "依存关系": "mark", "支配词 Head": "猎人"},
                {"序号": 4, "词": "狗", "词形还原": "狗", "词性 POS": "NOUN", "细粒度标签": "名词", "依存关系": "nsubj", "支配词 Head": "咬死了"},
            ]
        )
        core_args = pd.DataFrame(
            [
                {"词": "狗", "依存关系": "nsubj", "说明": "名词性主语", "支配词": "咬死了", "词性": "NOUN"},
                {"词": "咬死了", "依存关系": "ROOT", "说明": "句子根节点", "支配词": "ROOT", "词性": "VERB"},
                {"词": "猎人", "依存关系": "obj", "说明": "宾语", "支配词": "咬死了", "词性": "NOUN"},
            ]
        )
        observations = [
            {
                "侦探任务": "中文歧义：咬死了猎人的狗",
                "观察": "规则兜底当前展示“狗咬死了猎人”的读法；另一种读法是“某人咬死了猎人的狗”。两种结构会在成分树中同时给出。",
            }
        ]
        return ChineseSyntaxFallback(manual_dependency_svg(words, arcs), features, core_args, observations, "中文规则兜底：展示关系从句读法。")

    return None


def token_feature_table(doc: spacy.tokens.Doc) -> pd.DataFrame:
    """生成词级特征表，辅助观察 POS、依存关系和支配词。"""
    rows = []
    for token in doc:
        rows.append(
            {
                "序号": token.i + 1,
                "词": token.text,
                "词形还原": token.lemma_ or token.text,
                "词性 POS": token.pos_ or "-",
                "细粒度标签": token.tag_ or "-",
                "依存关系": token.dep_ or "-",
                "支配词 Head": token.head.text if token.head is not token else "ROOT",
            }
        )
    return pd.DataFrame(rows)


def core_argument_table(doc: spacy.tokens.Doc) -> pd.DataFrame:
    """抽取 nsubj / dobj / pobj / ROOT 等核心论元，展示句子骨架。"""
    rows = []
    for token in doc:
        dep = token.dep_ or "-"
        if dep in CORE_DEPS:
            rows.append(
                {
                    "词": token.text,
                    "依存关系": dep,
                    "说明": explain_dep(dep),
                    "支配词": token.head.text if token.head is not token else "ROOT",
                    "词性": token.pos_ or "-",
                }
            )
    return pd.DataFrame(rows)


def explain_dep(dep: str) -> str:
    explanations = {
        "nsubj": "名词性主语",
        "nsubjpass": "被动主语",
        "dobj": "直接宾语",
        "obj": "宾语",
        "pobj": "介词宾语",
        "ROOT": "句子根节点",
    }
    return explanations.get(dep, "核心成分")


def clean_words_for_cfg(text: str) -> list[str]:
    return [word.strip(".,!?;:\"'").lower() for word in text.split() if word.strip(".,!?;:\"'")]


def cfg_parser_for_sentence(words: list[str]) -> tuple[ChartParser, str]:
    """为课堂示例构造小型 CFG；它故意保留歧义，方便展示多棵成分树。"""
    if words == ["the", "boy", "saw", "the", "man", "with", "the", "telescope"]:
        grammar = CFG.fromstring(
            """
            S -> NP VP
            VP -> V NP | V NP PP
            NP -> Det N | NP PP
            PP -> P NP
            Det -> 'the'
            N -> 'boy' | 'man' | 'telescope'
            V -> 'saw'
            P -> 'with'
            """
        )
        note = "该 CFG 会产生两种结构：PP 附着到 NP，或 PP 附着到 VP。"
    elif words == ["fruit", "flies", "like", "a", "banana"]:
        grammar = CFG.fromstring(
            """
            S -> NP VP | N V PP
            NP -> Adj N | Det N
            VP -> V NP
            PP -> P NP
            Adj -> 'fruit'
            N -> 'fruit' | 'flies' | 'banana'
            V -> 'flies' | 'like'
            P -> 'like'
            Det -> 'a'
            """
        )
        note = "该 CFG 保留 flies=名词/动词、like=动词/介词的歧义。"
    else:
        grammar = CFG.fromstring(
            """
            S -> NP VP
            VP -> V NP | V NP PP
            NP -> Det N | N | NP PP
            PP -> P NP
            Det -> 'the' | 'a'
            N -> 'boy' | 'man' | 'telescope' | 'fruit' | 'flies' | 'banana'
            V -> 'saw' | 'like' | 'flies'
            P -> 'with' | 'like'
            """
        )
        note = "当前句子不在手写 CFG 的主覆盖范围内，若 benepar 不可用可能无法生成完整树。"
    return ChartParser(grammar), note


def chinese_constituency_trees(text: str) -> tuple[list[Tree], str]:
    """为中文歧义例句手写成分结构树，展示多种可能附着方式。"""
    compact_text = "".join(char for char in text if "\u4e00" <= char <= "\u9fff")
    if compact_text == "南京市长江大桥":
        trees = [
            Tree(
                "NP",
                [
                    Tree("NR-地名", ["南京市"]),
                    Tree("NP-桥名", [Tree("NR", ["长江"]), Tree("NN", ["大桥"])]),
                ],
            ),
            Tree(
                "NP-错误切分示例",
                [
                    Tree("NP", [Tree("NR", ["南京"]), Tree("NN", ["市长"])]),
                    Tree("NN", ["江大桥"]),
                ],
            ),
        ]
        return trees, "中文 CFG 兜底：展示“南京市 + 长江大桥”和一种错误切分结构，用来观察分词/句法耦合造成的歧义。"

    if compact_text == "咬死了猎人的狗":
        trees = [
            Tree(
                "S-读法A",
                [
                    Tree("VP", [Tree("V", ["咬死了"]), Tree("NP", [Tree("NP", ["猎人"]), Tree("DEG", ["的"]), Tree("NN", ["狗"])])]),
                ],
            ),
            Tree(
                "NP-读法B",
                [
                    Tree("CP", [Tree("VP", [Tree("V", ["咬死了"]), Tree("NP", ["猎人"])]), Tree("DEC", ["的"])]),
                    Tree("NN", ["狗"]),
                ],
            ),
        ]
        return trees, "中文 CFG 兜底：读法A 是“咬死了 [猎人的狗]”；读法B 是“[咬死了猎人] 的狗”。"

    return [], "中文成分句法暂未覆盖该句。可输入“南京市长江大桥”或“咬死了猎人的狗”观察手写 CFG 兜底树。"


def parse_constituency(text: str, is_chinese: bool) -> tuple[list[Tree], str, str]:
    """优先使用 benepar；失败或中文输入时使用 NLTK CFG fallback。"""
    if is_chinese:
        trees, status = chinese_constituency_trees(text)
        return trees, status, "中文 CFG"

    benepar_nlp, benepar_status = load_benepar_pipeline()
    if benepar_nlp is not None:
        try:
            doc = benepar_nlp(text)
            trees = [Tree.fromstring(sent._.parse_string) for sent in doc.sents if sent._.parse_string]
            if trees:
                return trees, f"{benepar_status} 当前展示 benepar 预训练模型输出。", "benepar"
        except Exception as exc:
            benepar_status = f"{benepar_status}\nbenepar 解析失败，改用 CFG fallback：{exc}"

    words = clean_words_for_cfg(text)
    parser, cfg_note = cfg_parser_for_sentence(words)
    try:
        trees = list(parser.parse(words))
    except ValueError as exc:
        return [], f"{benepar_status}\n{cfg_note}\nCFG 无法覆盖当前词表：{exc}", "NLTK CFG"
    return trees[:4], f"{benepar_status}\n{cfg_note}", "NLTK CFG"


def tree_to_nested_html(tree: Tree | str) -> str:
    """把 NLTK Tree 转成可折叠感较强的嵌套 HTML，避免依赖额外图形库。"""
    if isinstance(tree, str):
        return f'<span class="tree-leaf">{html.escape(tree)}</span>'
    children = "".join(tree_to_nested_html(child) for child in tree)
    return f'<div class="tree-node"><span>{html.escape(tree.label())}</span><div>{children}</div></div>'


def render_constituency_trees(trees: list[Tree]) -> None:
    if not trees:
        st.info("当前没有可展示的成分树。")
        return
    for index, tree in enumerate(trees, start=1):
        with st.expander(f"成分树 {index}", expanded=index == 1):
            st.markdown(f'<div class="constituency-tree">{tree_to_nested_html(tree)}</div>', unsafe_allow_html=True)
            st.code(tree.pformat(margin=90), language="text")


def render_knowledge_contrast() -> None:
    """展示依存句法和成分句法的知识对比，方便课堂讲解。"""
    st.markdown("## 句法知识对比")
    st.markdown(
        """
        <div class="syntax-compare-grid">
            <article>
                <span>Dependency Parsing</span>
                <h3>依存句法：谁支配谁</h3>
                <p>把句子看成词与词之间的有向关系图。每个词通常有一个支配词 Head，边标签表示主语、宾语、介词宾语、修饰语等语法关系。</p>
                <strong>适合观察：nsubj 主语、obj/dobj 宾语、pobj 介词宾语、ROOT 根节点。</strong>
            </article>
            <article>
                <span>Constituency Parsing</span>
                <h3>成分句法：短语如何嵌套</h3>
                <p>把句子看成短语层级树。NP、VP、PP 等短语不断组合成更大的短语，适合观察介词短语 PP 到底附着到 NP 还是 VP。</p>
                <strong>适合观察：NP 名词短语、VP 动词短语、PP 介词短语、S 句子整体结构。</strong>
            </article>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_attachment_hint(doc: spacy.tokens.Doc, text: str, extra: list[dict[str, str]] | None = None) -> None:
    """把歧义观察作为轻量提示放在对应模块内，而不是单独做成页面模块。"""
    observations = extra if extra is not None else attachment_observation(doc, text)
    if not observations:
        return
    content = " ".join(item["观察"] for item in observations)
    st.markdown(f'<div class="syntax-hint">{html.escape(content)}</div>', unsafe_allow_html=True)


def attachment_observation(doc: spacy.tokens.Doc, text: str) -> list[dict[str, str]]:
    """根据模型输出自动生成两个侦探任务的观察记录。"""
    observations = []
    lower_text = text.lower()

    if "telescope" in lower_text:
        telescope = next((token for token in doc if token.text.lower() == "telescope"), None)
        if telescope is not None:
            # telescope 通常是介词 with 的宾语；真正体现 PP 附着的是 with 的支配词。
            attachment = telescope.head.head if telescope.head.text.lower() == "with" else telescope.head
            head = attachment.text
            if head.lower() == "man":
                meaning = "模型倾向于解释为“拿着望远镜的男人”。"
            elif head.lower() == "saw":
                meaning = "模型倾向于解释为“男孩用望远镜看”。"
            else:
                meaning = "模型给出的附着点不是 man 或 saw，需要结合图中箭头进一步解释。"
            observations.append({"侦探任务": "望远镜歧义", "观察": f"with the telescope 附着到 {head}。{meaning}"})

    if lower_text.strip().startswith("fruit flies"):
        flies = next((token for token in doc if token.text.lower() == "flies"), None)
        if flies is not None:
            if flies.pos_ == "VERB":
                meaning = "模型把 flies 当成动词“飞”。"
            elif flies.pos_ in {"NOUN", "PROPN"}:
                meaning = "模型把 flies 当成名词“苍蝇”。"
            else:
                meaning = f"模型给出的 POS 是 {flies.pos_ or '-'}，需要人工判断是否被歧义句骗过。"
            observations.append({"侦探任务": "果蝇与香蕉", "观察": meaning})
    return observations


def render_syntax_app() -> None:
    st.markdown(
        """
        <section class="module-hero syntax-hero" style="--accent:#06b6d4">
            <p class="eyebrow">APP 02 · SYNTACTIC PARSING</p>
            <h1>句法分析应用</h1>
            <p>同时观察依存句法图与成分句法树，理解短语嵌套、词间支配关系以及经典结构歧义。</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    sample_name = st.selectbox("选择课堂示例", list(DEFAULT_SENTENCES))
    sentence = st.text_area(
        "输入一句话",
        value=DEFAULT_SENTENCES[sample_name],
        height=92,
        help="推荐先观察 The boy saw the man with the telescope. 和 Fruit flies like a banana.",
    )
    language_choice = st.radio("语言模型", ["自动识别", "英文", "中文"], horizontal=True)

    with st.spinner("正在运行句法分析模型..."):
        doc, resource = parse_dependency(sentence, language_choice)
        is_chinese = contains_chinese(sentence)
        chinese_fallback = chinese_syntax_fallback(sentence) if is_chinese and not resource.has_dependency_parser else None
        constituency_trees, constituency_status, constituency_engine = parse_constituency(sentence, is_chinese)

    status_cols = st.columns(3)
    status_cols[0].metric("spaCy 模型", resource.model_name)
    status_cols[1].metric("依存分析", "可用" if resource.has_dependency_parser else "降级")
    status_cols[2].metric("成分分析", constituency_engine)
    st.caption(resource.status)

    render_knowledge_contrast()

    dep_tab, const_tab = st.tabs(["依存关系", "成分结构"])
    with dep_tab:
        st.markdown("### 依存句法图")
        svg = chinese_fallback.svg if chinese_fallback else dependency_svg(doc)
        if svg:
            st.markdown(f'<div class="dependency-svg">{svg}</div>', unsafe_allow_html=True)
        else:
            st.warning("当前模型没有依存句法解析器，无法渲染 displaCy 依存图。可安装对应 spaCy 模型后重试。")

        if chinese_fallback:
            st.caption(chinese_fallback.status)

        render_attachment_hint(doc, sentence, chinese_fallback.observations if chinese_fallback else None)

        feature_df = chinese_fallback.features if chinese_fallback else token_feature_table(doc)
        core_df = chinese_fallback.core_args if chinese_fallback else core_argument_table(doc)
        feature_col, core_col = st.columns([1.45, 1], gap="large")
        with feature_col:
            st.markdown("### 词级特征表")
            st.dataframe(feature_df, use_container_width=True, hide_index=True)
        with core_col:
            st.markdown("### 核心论元提取器")
            if core_df.empty:
                st.warning("没有抽取到 nsubj / dobj / obj / pobj / ROOT。")
            else:
                st.dataframe(core_df, use_container_width=True, hide_index=True)

    with const_tab:
        st.markdown("### 成分句法树")
        st.caption(constituency_status)
        render_constituency_trees(constituency_trees)
