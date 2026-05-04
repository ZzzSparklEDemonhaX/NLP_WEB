from __future__ import annotations

import html
import re

import pandas as pd
import streamlit as st
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


MODEL_NAME = "Helsinki-NLP/opus-mt-en-zh"
DEFAULT_SOURCE = "It rains cats and dogs."
DEFAULT_REFERENCE = "下着倾盆大雨。"

RULE_DICT = {
    "it": "它",
    "rains": "下雨",
    "rain": "雨",
    "cats": "猫",
    "and": "和",
    "dogs": "狗",
    "the": "这",
    "man": "男人",
    "went": "去了",
    "go": "去",
    "to": "到",
    "buy": "买",
    "some": "一些",
    "milk": "牛奶",
    "apple": "苹果",
    "is": "是",
    "company": "公司",
    "in": "在",
    "china": "中国",
    "language": "语言",
    "model": "模型",
    "machine": "机器",
    "translation": "翻译",
}

PHRASE_FALLBACK = {
    "it rains cats and dogs.": "下着倾盆大雨。",
    "the man went to the store to buy some milk.": "那个男人去商店买了一些牛奶。",
    "machine translation is useful.": "机器翻译很有用。",
}

IDIOM_REWRITES = {
    r"\bit rains cats and dogs\b": "it is raining very heavily",
    r"\bkick the bucket\b": "die",
    r"\bpiece of cake\b": "very easy",
    r"\bbreak the ice\b": "start the conversation",
    r"\bspill the beans\b": "reveal the secret",
}


@st.cache_resource(show_spinner=False)
def load_translation_pipeline():
    """Load the local English-to-Chinese translation model."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, local_files_only=True)
    model.eval()
    return pipeline("translation", model=model, tokenizer=tokenizer, device=-1)


def short_error(exc: Exception) -> str:
    message = str(exc).replace("\n", " ").strip()
    if len(message) > 120:
        return message[:120] + "..."
    return message or exc.__class__.__name__


def rewrite_idioms_for_nmt(text: str) -> tuple[str, str]:
    """Rewrite common idioms into plain English so the lightweight MT model sees sentence meaning."""
    rewritten = text
    notes = []
    for pattern, replacement in IDIOM_REWRITES.items():
        if re.search(pattern, rewritten, flags=re.IGNORECASE):
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
            notes.append(replacement)
    return rewritten, "；".join(notes)


def clean_translation(text: str) -> str:
    """Clean obvious repetition artifacts from the small local model."""
    normalized = text.strip()
    if "，" in normalized:
        parts = [part.strip() for part in normalized.split("，") if part.strip()]
        if len(parts) == 2 and parts[0] == parts[1]:
            return parts[0]
    return normalized


def nmt_translate(text: str) -> tuple[str, str]:
    """Translate a sentence with phrase-aware preprocessing and local NMT inference."""
    try:
        translator = load_translation_pipeline()
        rewritten, note = rewrite_idioms_for_nmt(text)
        result = translator(rewritten, max_length=128)
        translation = clean_translation(result[0]["translation_text"])
        status = "已加载 Helsinki-NLP/opus-mt-en-zh 本地缓存模型。"
        if note:
            status += f" 已对习语/短语做语义改写：{note}"
        return translation, status
    except Exception as exc:
        fallback = PHRASE_FALLBACK.get(text.strip().lower(), rule_based_translate(text))
        return fallback, f"NMT 模型加载失败，已使用兜底译文：{short_error(exc)}"


def rule_based_translate(text: str) -> str:
    """Simulate early rule-based MT with token-by-token substitution."""
    tokens = re.findall(r"[A-Za-z']+|[^\w\s]", text.lower())
    translated = []
    for token in tokens:
        if re.fullmatch(r"[^\w\s]", token):
            translated.append(token)
        else:
            translated.append(RULE_DICT.get(token, token))
    return " ".join(translated).replace(" .", "。").replace(" ,", "，")


def chinese_bleu(reference: str, candidate: str) -> float:
    """Compute a simple character-level BLEU score for Chinese demo text."""
    reference_tokens = [char for char in reference.strip() if not char.isspace()]
    candidate_tokens = [char for char in candidate.strip() if not char.isspace()]
    if not reference_tokens or not candidate_tokens:
        return 0.0
    smoothing = SmoothingFunction().method1
    return float(sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing))


def render_mt_intro() -> None:
    st.markdown(
        """
        <section class="module-hero mt-hero" style="--accent:#f97316">
            <p class="eyebrow">APP 08 / MACHINE TRANSLATION</p>
            <h1>机器翻译对比与评测应用</h1>
            <p>对比规则直译和神经机器翻译，并通过 BLEU 观察候选译文与参考译文之间的 n-gram 匹配程度。</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_nmt_tab() -> None:
    st.markdown("### 模块 1：神经机器翻译引擎 NMT")
    text = st.text_area("输入英文句子", value=DEFAULT_SOURCE, height=110)
    if st.button("运行神经机器翻译", use_container_width=True):
        with st.spinner("NMT 模型正在翻译..."):
            translation, status = nmt_translate(text)
        st.caption(status)
        st.markdown(
            f'<div class="translation-card"><span>NMT Translation</span><p>{html.escape(translation)}</p></div>',
            unsafe_allow_html=True,
        )
    st.info("观察：习语和固定短语若直接逐词翻译往往失真，因此这里会先做短语级语义改写，再送入 NMT 模型。")


def render_compare_tab() -> None:
    st.markdown("### 模块 2：基于规则的直译 vs 神经网络意译")
    text = st.text_area("输入英文句子进行对比", value=DEFAULT_SOURCE, height=100, key="compare_source")
    literal = rule_based_translate(text)
    with st.spinner("正在生成 NMT 译文..."):
        nmt_text, status = nmt_translate(text)
    st.caption(status)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            f'<div class="translation-card literal"><span>Rule-based 逐词直译</span><p>{html.escape(literal)}</p></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div class="translation-card nmt"><span>NMT 神经翻译</span><p>{html.escape(nmt_text)}</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="mt-note">规则直译几乎不处理语序、上下文和习语；这里的 NMT 会先做短语语义归一化，再交给句级模型生成译文，因此比逐词替换更接近真实语义。</div>',
        unsafe_allow_html=True,
    )


def render_bleu_tab() -> None:
    st.markdown("### 模块 3：机器翻译质量自动评测 BLEU")
    source = st.text_area("待翻译英文原文", value=DEFAULT_SOURCE, height=80)
    with st.spinner("正在准备候选译文..."):
        generated, status = nmt_translate(source)
    reference = st.text_area("标准中文参考译文 Reference", value=DEFAULT_REFERENCE, height=80)
    candidate = st.text_area("机器生成候选译文 Candidate", value=generated, height=80)
    score = chinese_bleu(reference, candidate)

    st.caption(status)
    st.metric("BLEU Score", f"{score:.4f}")
    st.dataframe(
        pd.DataFrame(
            [
                {"类型": "Source", "文本": source},
                {"类型": "Reference", "文本": reference},
                {"类型": "Candidate", "文本": candidate},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(
        '<div class="mt-note">BLEU 基于 n-gram 匹配，分数越高表示候选译文与参考译文在表层片段上更接近；它适合做自动对比，但不能完全替代理解层面的人工评价。</div>',
        unsafe_allow_html=True,
    )


def render_machine_translation_app() -> None:
    render_mt_intro()
    tab1, tab2, tab3 = st.tabs(["NMT 引擎", "规则直译 vs NMT", "BLEU 评测"])
    with tab1:
        try:
            render_nmt_tab()
        except Exception as exc:
            st.error(f"NMT 模块运行失败：{short_error(exc)}")
    with tab2:
        try:
            render_compare_tab()
        except Exception as exc:
            st.error(f"对比模块运行失败：{short_error(exc)}")
    with tab3:
        try:
            render_bleu_tab()
        except Exception as exc:
            st.error(f"BLEU 模块运行失败：{short_error(exc)}")
