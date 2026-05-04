from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from gensim.models import FastText, Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.utils import simple_preprocess
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 默认语料控制在课堂实验可快速训练的规模。用户可以直接替换为维基百科、文学作品等 500-1000 词英文文本。
DEFAULT_CORPUS = """
Natural language processing studies how computers represent, understand, and generate human language.
Words that appear in similar contexts often share semantic properties, and distributional models try to
capture this regularity with vectors. A small story about science can still reveal useful patterns. A
researcher reads a book about language, while a student writes notes about models, data, words, and meaning.
The researcher explains that a vector space can place related words near each other. In this space, language
and text may be close, science and research may be close, and computer and machine may be close. Traditional
models such as TF IDF represent each document with sparse keyword weights. Latent semantic analysis then
factorizes a term document matrix and compresses it into a dense low dimensional representation. Neural
models such as Word2Vec learn vectors by predicting words from context or predicting context from words.
CBOW predicts a target word from surrounding context words, while Skip Gram predicts surrounding words from
one target word. Global vector models such as GloVe use co occurrence statistics from a large corpus. FastText
adds subword features, so it can produce vectors for misspelled or rare words such as computeer or langauge.
Sentence vectors can be estimated by averaging word vectors across a sentence. This simple pooling method is
not perfect, but it gives students a clear view of how local word meaning can become a global sentence
representation. A machine can read text, compare meaning, search documents, and retrieve similar words.
Language models, vector models, and semantic analysis together form a practical foundation for modern NLP.
"""


@dataclass(frozen=True)
class CorpusBundle:
    documents: list[str]
    tokenized_sentences: list[list[str]]
    vocabulary: list[str]


def split_documents(text: str) -> list[str]:
    """把英文长文本按句子切成文档集合；不用依赖 NLTK punkt，避免首次运行缺数据。"""
    docs = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    return docs or [text.strip()]


def tokenize_sentence(sentence: str) -> list[str]:
    """使用 gensim.simple_preprocess 做英文分词、大小写归一化和基础清洗。"""
    return simple_preprocess(sentence, deacc=True, min_len=2)


def build_corpus_bundle(text: str) -> CorpusBundle:
    """把原始语料转换成四个模块共用的数据结构。"""
    documents = split_documents(text)
    tokenized = [tokenize_sentence(doc) for doc in documents]
    tokenized = [tokens for tokens in tokenized if tokens]
    vocabulary = sorted({token for sent in tokenized for token in sent})
    return CorpusBundle(documents=documents, tokenized_sentences=tokenized, vocabulary=vocabulary)


def safe_top_terms(feature_names: np.ndarray, matrix, top_n: int = 5) -> pd.DataFrame:
    """从 TF-IDF 矩阵中提取全局最高权重词，用于关键词展示。"""
    if matrix.shape[1] == 0:
        return pd.DataFrame(columns=["关键词", "最高 TF-IDF 权重"])
    max_scores = matrix.max(axis=0).toarray().ravel().astype(float)
    top_indices = max_scores.argsort()[::-1][:top_n]
    return pd.DataFrame(
        {
            "关键词": feature_names[top_indices].astype(str),
            "最高 TF-IDF 权重": np.round(max_scores[top_indices], 4).astype(float),
        }
    )


def compute_lsa_coordinates(documents: list[str], use_tfidf: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算 TF-IDF/Count 矩阵，并对词项矩阵做 SVD，得到词汇二维坐标。"""
    if not documents:
        raise ValueError("请至少输入一个英文句子。")
    vectorizer_cls = TfidfVectorizer if use_tfidf else CountVectorizer
    vectorizer = vectorizer_cls(stop_words="english", min_df=1)
    matrix = vectorizer.fit_transform(documents)
    if matrix.shape[1] == 0:
        raise ValueError("去除停用词后词表为空，请输入更多英文内容。")
    feature_names = vectorizer.get_feature_names_out()

    # 词向量视角：把 document-term 矩阵转置为 term-document 矩阵，再做 TruncatedSVD。
    term_doc = matrix.T
    n_components = 2 if min(term_doc.shape) >= 3 else 1
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    coords = svd.fit_transform(term_doc)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])

    coord_df = pd.DataFrame({"word": feature_names, "x": coords[:, 0], "y": coords[:, 1]})
    matrix_df = pd.DataFrame(matrix.toarray(), columns=feature_names)
    matrix_df.insert(0, "doc_id", [f"Doc {index + 1}" for index in range(matrix.shape[0])])
    return matrix_df, coord_df


@st.cache_resource(show_spinner=False)
def train_word2vec(sentences_key: str, sg: int, window: int, vector_size: int, epochs: int) -> Word2Vec:
    """训练 Word2Vec；用字符串 key 让 Streamlit 能缓存结果。"""
    sentences = [line.split() for line in sentences_key.split("\n") if line.strip()]
    return Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=1,
        sg=sg,
        epochs=epochs,
        seed=42,
    )


@st.cache_resource(show_spinner=False)
def train_fasttext(sentences_key: str, window: int, vector_size: int, epochs: int) -> FastText:
    """训练 FastText；min_n/max_n 启用字符 n-gram，从而处理 OOV 单词。"""
    sentences = [line.split() for line in sentences_key.split("\n") if line.strip()]
    return FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=1,
        min_n=3,
        max_n=6,
        epochs=epochs,
        seed=42,
    )


def sentences_to_key(sentences: list[list[str]]) -> str:
    """把 tokenized sentences 转为稳定字符串，作为模型缓存 key。"""
    return "\n".join(" ".join(sentence) for sentence in sentences)


def cosine_for_words(model: Word2Vec | FastText | KeyedVectors, word_a: str, word_b: str) -> float | None:
    """计算两个词的余弦相似度；词不存在时返回 None。"""
    try:
        return float(model.wv.similarity(word_a, word_b)) if hasattr(model, "wv") else float(model.similarity(word_a, word_b))
    except KeyError:
        return None


def average_sentence_vector(model: FastText, sentence: str) -> np.ndarray | None:
    """Sent2Vec 简化实现：句向量 = 句中所有词向量的平均池化。"""
    tokens = tokenize_sentence(sentence)
    if not tokens:
        return None
    vectors = [model.wv[token] for token in tokens]
    return np.mean(vectors, axis=0)


def vector_cosine(vec_a: np.ndarray | None, vec_b: np.ndarray | None) -> float | None:
    """计算两个句向量之间的余弦相似度。"""
    if vec_a is None or vec_b is None:
        return None
    denominator = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denominator == 0:
        return None
    return float(np.dot(vec_a, vec_b) / denominator)


def local_glove_fallback() -> KeyedVectors:
    """构造一个小型教学 KeyedVectors，保证无网络时词类比模块仍可展示公式效果。"""
    vectors = {
        "man": [1, 0, 0, 0, 0, 0],
        "woman": [1, 1, 0, 0, 0, 0],
        "king": [1, 0, 1, 0, 0, 0],
        "queen": [1, 1, 1, 0, 0, 0],
        "france": [0, 0, 0, 0, 1, 0],
        "paris": [0, 0, 0, 1, 1, 0],
        "china": [0, 0, 0, 0, 2, 0],
        "beijing": [0, 0, 0, 1, 2, 0],
        "japan": [0, 0, 0, 0, 3, 0],
        "tokyo": [0, 0, 0, 1, 3, 0],
        "good": [0, 0, 0, 0, 0, 1],
        "great": [0, 0, 0, 0, 0, 0.95],
    }
    keyed_vectors = KeyedVectors(vector_size=6)
    keyed_vectors.add_vectors(list(vectors), np.asarray(list(vectors.values()), dtype=np.float32))
    return keyed_vectors


@st.cache_resource(show_spinner=False)
def load_glove_model(try_real_model: bool) -> tuple[KeyedVectors, str]:
    """优先加载 glove-twitter-25；失败时退回本地教学向量，避免网络影响展示。"""
    if try_real_model:
        try:
            import gensim.downloader as api

            model = api.load("glove-twitter-25")
            return model, "已加载 gensim.downloader 的 glove-twitter-25 预训练模型。"
        except Exception as exc:
            return local_glove_fallback(), f"真实 GloVe 加载失败，已启用本地教学 fallback：{short_error(exc)}"
    return local_glove_fallback(), "当前使用本地教学 fallback。勾选“尝试加载真实 GloVe”可调用 gensim.downloader。"


def short_error(exc: Exception) -> str:
    """压缩错误信息，避免把 traceback 展示到课堂页面。"""
    message = str(exc).replace("\n", " ").strip()
    if len(message) > 120:
        return message[:120] + "..."
    return message or exc.__class__.__name__


def render_semantic_intro() -> None:
    st.markdown(
        """
        <section class="module-hero semantic-hero" style="--accent:#22c55e">
            <p class="eyebrow">APP 03 · SEMANTIC REPRESENTATION</p>
            <h1>语义分析空间</h1>
            <p>集成 TF-IDF、LSA、Word2Vec、GloVe、FastText 与句向量实验，观察从统计表示到神经表示的语义建模过程。</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_theory_strip() -> None:
    st.markdown(
        """
        <div class="semantic-theory-grid">
            <article><span>TF-IDF</span><p>用词频与逆文档频率衡量关键词重要性，强调“本文档高频、全局低频”的词。</p></article>
            <article><span>LSA</span><p>通过 SVD 将高维稀疏矩阵压缩为低维稠密空间，捕捉潜在语义关联。</p></article>
            <article><span>Word2Vec</span><p>CBOW 用上下文预测目标词，Skip-Gram 用目标词预测上下文。</p></article>
            <article><span>FastText</span><p>引入字符 n-gram 子词特征，对拼写错误和 OOV 单词更鲁棒。</p></article>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tfidf_lsa_tab(bundle: CorpusBundle) -> None:
    st.markdown("### 模块 1：传统统计模型 TF-IDF 与 LSA")
    use_tfidf = st.radio("LSA 输入矩阵", ["TF-IDF", "One-hot / Count"], horizontal=True) == "TF-IDF"
    try:
        matrix_df, coord_df = compute_lsa_coordinates(bundle.documents, use_tfidf)
    except ValueError as exc:
        st.error(f"语料太短或词表为空，无法计算矩阵：{exc}")
        return

    st.markdown("#### TF-IDF / Count 矩阵")
    st.dataframe(matrix_df.round(4), use_container_width=True, hide_index=True)

    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    tfidf_matrix = vectorizer.fit_transform(bundle.documents)
    keyword_df = safe_top_terms(vectorizer.get_feature_names_out(), tfidf_matrix)
    left, right = st.columns([0.85, 1.4], gap="large")
    with left:
        st.markdown("#### Top 5 关键词")
        st.dataframe(keyword_df, use_container_width=True, hide_index=True)
        st.markdown(
            '<div class="semantic-formula">TF-IDF(t,d) = TF(t,d) × IDF(t)</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("#### LSA 词汇二维空间")
        top_words = coord_df.head(80)
        fig = px.scatter(top_words, x="x", y="y", text="word", color="x", color_continuous_scale="Greens")
        fig.update_traces(textposition="top center")
        fig.update_layout(height=430, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.caption("观察任务：如果两个词在语料中经常同现，LSA 的二维坐标可能更接近；小语料下这种趋势会比较粗糙。")


def render_word2vec_tab(bundle: CorpusBundle) -> Word2Vec | None:
    st.markdown("### 模块 2：Word2Vec 训练与对比")
    if not bundle.tokenized_sentences:
        st.error("语料没有可训练的英文 token。")
        return None

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        architecture = st.radio("训练架构", ["CBOW", "Skip-Gram"], horizontal=False)
    with col_b:
        window = st.slider("window 上下文窗口", 2, 10, 5)
    with col_c:
        vector_size = st.slider("向量维度", 20, 120, 50, step=10)
    with col_d:
        epochs = st.slider("训练轮数", 20, 200, 80, step=20)

    sg = 0 if architecture == "CBOW" else 1
    model = train_word2vec(sentences_to_key(bundle.tokenized_sentences), sg, window, vector_size, epochs)

    vocab = sorted(model.wv.index_to_key)
    default_word = "language" if "language" in vocab else vocab[0]
    query = st.text_input("输入目标词，查询 Top 5 相似词", value=default_word).strip().lower()
    st.markdown(
        '<div class="semantic-formula">sim(wᵢ,wⱼ)=cos(vᵢ,vⱼ)=vᵢ·vⱼ/(||vᵢ||||vⱼ||)</div>',
        unsafe_allow_html=True,
    )
    if query not in model.wv:
        st.warning(f"`{query}` 不在当前实时训练语料的词表中。")
    else:
        similar = model.wv.most_similar(query, topn=min(5, len(vocab) - 1))
        st.dataframe(pd.DataFrame(similar, columns=["相似词", "余弦相似度"]), use_container_width=True, hide_index=True)

    with st.expander("查看当前 Word2Vec 词表"):
        st.write(", ".join(vocab[:200]))
    return model


def render_glove_tab() -> None:
    st.markdown("### 模块 3：预训练 GloVe 与词类比")
    try_real_model = st.checkbox("尝试加载真实 glove-twitter-25（需要网络或本地缓存）", value=False)
    with st.spinner("正在准备 GloVe / 教学向量..."):
        glove_model, status = load_glove_model(try_real_model)
    st.caption(status)

    st.markdown("#### 词类比计算器：A - B + C")
    col_a, col_b, col_c = st.columns(3)
    a = col_a.text_input("A", value="king").strip().lower()
    b = col_b.text_input("B", value="man").strip().lower()
    c = col_c.text_input("C", value="woman").strip().lower()
    st.markdown('<div class="semantic-formula">Result = Vector(A) - Vector(B) + Vector(C)</div>', unsafe_allow_html=True)

    try:
        results = glove_model.most_similar(positive=[a, c], negative=[b], topn=5)
        st.dataframe(pd.DataFrame(results, columns=["预测词", "相似度"]), use_container_width=True, hide_index=True)
    except KeyError as exc:
        st.warning(f"词表中缺少：{exc}")

    st.markdown("#### 两词语义相似度")
    sim_a, sim_b = st.columns(2)
    word_a = sim_a.text_input("单词 1", value="good").strip().lower()
    word_b = sim_b.text_input("单词 2", value="great").strip().lower()
    try:
        score = float(glove_model.similarity(word_a, word_b))
        st.metric("Cosine Similarity", f"{score:.4f}")
    except KeyError as exc:
        st.warning(f"词表中缺少：{exc}")

    st.info("可尝试：king - man + woman；paris - france + china；tokyo - japan + france。")


def render_fasttext_sent2vec_tab(bundle: CorpusBundle, word2vec_model: Word2Vec | None) -> None:
    st.markdown("### 模块 4：FastText 子词特征与 Sent2Vec")
    if not bundle.tokenized_sentences:
        st.error("语料没有可训练的英文 token。")
        return

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        window = st.slider("FastText window", 2, 10, 5)
    with col_b:
        vector_size = st.slider("FastText 向量维度", 20, 120, 50, step=10)
    with col_c:
        epochs = st.slider("FastText 训练轮数", 20, 200, 80, step=20)

    ft_model = train_fasttext(sentences_to_key(bundle.tokenized_sentences), window, vector_size, epochs)

    st.markdown("#### OOV 测试")
    oov_word = st.text_input("输入拼写错误或未登录词", value="computeer").strip().lower()
    oov_cols = st.columns(2, gap="large")
    with oov_cols[0]:
        st.markdown("##### Word2Vec")
        if word2vec_model is None:
            st.info("请先在 Word2Vec 标签页训练模型。")
        else:
            try:
                _ = word2vec_model.wv[oov_word]
                st.success("Word2Vec 找到了该词向量。")
            except KeyError:
                st.error("未登录词：Word2Vec 无法直接给出向量。")
    with oov_cols[1]:
        st.markdown("##### FastText")
        try:
            _ = ft_model.wv[oov_word]
            similar = ft_model.wv.most_similar(oov_word, topn=5)
            st.success("FastText 通过字符 n-gram 成功生成 OOV 向量。")
            st.dataframe(pd.DataFrame(similar, columns=["相似词", "相似度"]), use_container_width=True, hide_index=True)
        except KeyError:
            st.warning("FastText 当前也无法处理该词，可能语料过小或字符特征不足。")

    st.markdown("#### Sent2Vec 简化实验：平均池化句向量")
    s1 = st.text_area("句子 1", value="A researcher studies language models and semantic vector spaces.", height=78)
    s2 = st.text_area("句子 2", value="A student learns how word vectors represent meaning in NLP.", height=78)
    score = vector_cosine(average_sentence_vector(ft_model, s1), average_sentence_vector(ft_model, s2))
    if score is None or math.isnan(score):
        st.warning("句子为空或向量无法计算。")
    else:
        st.metric("句向量余弦相似度", f"{score:.4f}")
    st.markdown('<div class="semantic-formula">vₛ = 1 / |R(s)| · Σ vᵥ，其中 R(s) 是句中词和 n-gram 特征集合</div>', unsafe_allow_html=True)


def render_semantic_app() -> None:
    render_semantic_intro()
    st.markdown("### 实验语料")
    corpus_text = st.text_area(
        "输入 500-1000 词英文语料",
        value=DEFAULT_CORPUS.strip(),
        height=210,
        help="四个模块会共用这段语料。建议包含重复主题词，便于观察向量空间中的聚类和相似性。",
    )
    bundle = build_corpus_bundle(corpus_text)

    metric_cols = st.columns(4)
    metric_cols[0].metric("文档/句子数", len(bundle.documents))
    metric_cols[1].metric("训练句数", len(bundle.tokenized_sentences))
    metric_cols[2].metric("词表大小", len(bundle.vocabulary))
    metric_cols[3].metric("总 token 数", sum(len(sent) for sent in bundle.tokenized_sentences))

    render_theory_strip()

    tab1, tab2, tab3, tab4 = st.tabs(["TF-IDF 与 LSA", "Word2Vec", "GloVe 类比", "FastText & Sent2Vec"])
    word2vec_model = None
    with tab1:
        try:
            render_tfidf_lsa_tab(bundle)
        except Exception as exc:
            st.error(f"TF-IDF / LSA 模块运行失败：{short_error(exc)}")
    with tab2:
        try:
            word2vec_model = render_word2vec_tab(bundle)
        except Exception as exc:
            st.error(f"Word2Vec 模块运行失败：{short_error(exc)}")
    with tab3:
        try:
            render_glove_tab()
        except Exception as exc:
            st.error(f"GloVe 模块运行失败：{short_error(exc)}")
    with tab4:
        # Streamlit 会顺序执行各标签页代码，因此这里可以复用上一标签训练出的 Word2Vec 模型。
        try:
            render_fasttext_sent2vec_tab(bundle, word2vec_model)
        except Exception as exc:
            st.error(f"FastText / Sent2Vec 模块运行失败：{short_error(exc)}")
