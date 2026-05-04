from __future__ import annotations

import html
import math
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd
import streamlit as st
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer


DEFAULT_NGRAM_CORPUS = """
The cat sat on the mat. The dog sat on the rug. The cat chased the mouse.
The mouse ran into the house. The dog chased the cat. A language model learns
patterns from text and predicts the next word from context.
"""

DEFAULT_RNN_CORPUS = "hello world hello world hello world. language models learn patterns. "


@dataclass(frozen=True)
class NgramModel:
    unigram_counts: Counter
    bigram_counts: Counter
    trigram_counts: Counter
    vocabulary: set[str]


class CharRNN(nn.Module):
    """最小可运行的字符级 RNN 语言模型。"""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        return self.fc(output), hidden


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def build_ngram_model(corpus: str) -> NgramModel:
    sentences = [tokenize_words(part) for part in re.split(r"[.!?]+", corpus) if tokenize_words(part)]
    unigram_counts: Counter = Counter()
    bigram_counts: Counter = Counter()
    trigram_counts: Counter = Counter()
    vocabulary = {"<s>", "</s>"}

    for sent in sentences:
        tokens = ["<s>", "<s>", *sent, "</s>"]
        vocabulary.update(tokens)
        unigram_counts.update(tokens)
        bigram_counts.update(zip(tokens[:-1], tokens[1:]))
        trigram_counts.update(zip(tokens[:-2], tokens[1:-1], tokens[2:]))

    return NgramModel(unigram_counts, bigram_counts, trigram_counts, vocabulary)


def trigram_probability(model: NgramModel, sentence: str, smoothing: bool) -> tuple[float, pd.DataFrame]:
    tokens = ["<s>", "<s>", *tokenize_words(sentence), "</s>"]
    vocab_size = max(len(model.vocabulary), 1)
    probability = 1.0
    rows = []

    for w1, w2, w3 in zip(tokens[:-2], tokens[1:-1], tokens[2:]):
        trigram_count = model.trigram_counts[(w1, w2, w3)]
        context_count = model.bigram_counts[(w1, w2)]
        if smoothing:
            step_prob = (trigram_count + 1) / (context_count + vocab_size)
            formula = f"({trigram_count}+1) / ({context_count}+|V|={vocab_size})"
        else:
            step_prob = 0.0 if context_count == 0 else trigram_count / context_count
            formula = f"{trigram_count} / {context_count}" if context_count else "上下文未出现 => 0"
        probability *= step_prob
        rows.append(
            {
                "三元组": f"({w1}, {w2}, {w3})",
                "三元组计数": trigram_count,
                "上下文计数": context_count,
                "条件概率 P(w_i | w_i-2,w_i-1)": step_prob,
                "计算公式": formula,
            }
        )
    return probability, pd.DataFrame(rows)


def summarize_joint_probability(probability: float, table: pd.DataFrame) -> tuple[float, float, str]:
    step_probs = table["条件概率 P(w_i | w_i-2,w_i-1)"].tolist()
    if not step_probs:
        return 0.0, float("-inf"), ""
    avg_step_prob = probability ** (1 / len(step_probs)) if probability > 0 else 0.0
    log_prob = math.log(probability) if probability > 0 else float("-inf")
    chain = " × ".join(f"{value:.3f}" for value in step_probs)
    return avg_step_prob, log_prob, chain


def prepare_char_data(text: str):
    cleaned = text if len(text) >= 4 else DEFAULT_RNN_CORPUS
    chars = sorted(set(cleaned))
    stoi = {char: index for index, char in enumerate(chars)}
    itos = {index: char for char, index in stoi.items()}
    encoded = torch.tensor([stoi[char] for char in cleaned], dtype=torch.long)
    x = encoded[:-1].unsqueeze(0)
    y = encoded[1:].unsqueeze(0)
    return x, y, stoi, itos


def train_char_rnn(corpus: str, hidden_size: int, epochs: int, learning_rate: float, progress_callback=None):
    torch.manual_seed(42)
    x, y, stoi, itos = prepare_char_data(corpus)
    model = CharRNN(len(stoi), hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses = []

    for _ in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, len(stoi)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if progress_callback is not None and (
            len(losses) == 1 or len(losses) % max(1, epochs // 20) == 0 or len(losses) == epochs
        ):
            progress_callback(losses)
    return model, stoi, itos, losses


def generate_text(model: CharRNN, stoi: dict[str, int], itos: dict[int, str], seed: str, length: int = 80) -> str:
    model.eval()
    if not seed:
        seed = next(iter(stoi))
    current_char = seed[-1] if seed[-1] in stoi else next(iter(stoi))
    generated = seed
    hidden = None
    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([[stoi[current_char]]], dtype=torch.long)
            logits, hidden = model(x, hidden)
            probs = torch.softmax(logits[0, -1], dim=0)
            next_index = int(torch.multinomial(probs, 1).item())
            current_char = itos[next_index]
            generated += current_char
    return generated


@st.cache_resource(show_spinner=False)
def load_mlm():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased", local_files_only=True)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def load_gpt2():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("gpt2", local_files_only=True)
    model.eval()
    return tokenizer, model


def predict_mask(sentence: str, top_k: int = 5) -> pd.DataFrame:
    tokenizer, model = load_mlm()
    if tokenizer.mask_token not in sentence:
        return pd.DataFrame({"候选词": ["请在句子中加入 [MASK]"], "概率": [0.0]})
    inputs = tokenizer(sentence, return_tensors="pt")
    mask_positions = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]
    if len(mask_positions) == 0:
        return pd.DataFrame({"候选词": ["未找到 [MASK]"], "概率": [0.0]})
    with torch.no_grad():
        logits = model(**inputs).logits[0, mask_positions[0]]
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, top_k)
    tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
    return pd.DataFrame({"候选词": tokens, "概率": values.detach().cpu().numpy().round(4)})


def generate_gpt2(prompt: str, max_new_tokens: int = 35) -> str:
    tokenizer, model = load_gpt2()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=40,
            top_p=0.92,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def sentence_perplexity(sentence: str) -> tuple[float, float]:
    tokenizer, model = load_gpt2()
    inputs = tokenizer(sentence, return_tensors="pt")
    if inputs["input_ids"].shape[1] < 2:
        return float("nan"), float("nan")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = float(outputs.loss.detach().cpu())
    return loss, math.exp(loss)


def render_lm_intro() -> None:
    st.markdown(
        """
        <section class="module-hero lm-hero" style="--accent:#14b8a6">
            <p class="eyebrow">APP 06 · LANGUAGE MODELING</p>
            <h1>语言模型训练与对比分析平台</h1>
            <p>
                从 n-gram 统计语言模型、加一平滑、字符级 RNN，自然过渡到 BERT、GPT-2
                与困惑度评测，帮助展示“统计建模”和“预训练生成”两条技术路线的差异。
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_ngram_tab() -> None:
    st.markdown("### 模块 1：n 元语言模型与数据平滑")
    st.info(
        "三元语言模型使用链式法则计算句子联合概率："
        "P(w1...wn)=Π P(w_i | w_i-2, w_i-1)。"
        "因为它是多个小于 1 的条件概率连乘，所以最终数值通常会比较小。"
    )
    corpus = st.text_area("训练语料", value=DEFAULT_NGRAM_CORPUS.strip(), height=150)
    sentence = st.text_input("测试句子", value="The cat sat on the mat.")
    model = build_ngram_model(corpus)
    smooth = st.checkbox("开启加一平滑 Add-one / Laplace Smoothing", value=False)

    prob, table = trigram_probability(model, sentence, smooth)
    avg_step_prob, log_prob, chain = summarize_joint_probability(prob, table)

    metric_cols = st.columns(3)
    metric_cols[0].metric("词表大小 |V|", len(model.vocabulary))
    metric_cols[1].metric("Trigram 数量", len(model.trigram_counts))
    metric_cols[2].metric("句子联合概率", f"{prob:.3e}")

    st.dataframe(table, use_container_width=True, hide_index=True)

    log_prob_text = f"{log_prob:.4f}" if math.isfinite(log_prob) else "-inf"
    st.markdown(
        f"""
        <div class="lm-formula">
            联合概率公式：P(w1...wn) = Π P(w_i | w_i-2, w_i-1)<br/>
            本句逐步连乘：{chain}<br/>
            长度归一化后的平均条件概率：{avg_step_prob:.4f}<br/>
            log Joint Probability：{log_prob_text}<br/>
            说明：像 5.556e-02 这样的结果并不表示公式有错，而是因为多个条件概率连续相乘后自然变小。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="lm-formula">未平滑：P = count(trigram) / count(context)。加一平滑：P = (count + 1) / (context_count + |V|)。</div>',
        unsafe_allow_html=True,
    )


def render_rnn_tab() -> None:
    st.markdown("### 模块 2：从零训练 RNN 语言模型")
    st.info(
        "训练逻辑：把字符序列编码成 ID，用前 t 个字符预测第 t+1 个字符；"
        "RNN 的隐藏状态负责保存前文信息，因此能学习局部模式。"
    )
    corpus = st.text_area("RNN 训练语料", value=DEFAULT_RNN_CORPUS * 3, height=135)
    col_a, col_b, col_c = st.columns(3)
    hidden_size = col_a.slider("Hidden Size", 16, 128, 48, step=16)
    epochs = col_b.slider("Epochs", 10, 200, 80, step=10)
    learning_rate = col_c.slider("Learning Rate", 0.001, 0.1, 0.02, step=0.001, format="%.3f")

    if st.button("开始训练 RNN", use_container_width=True):
        chart_box = st.empty()
        progress_bar = st.progress(0)

        def update_training_view(losses: list[float]) -> None:
            progress_bar.progress(min(len(losses) / epochs, 1.0))
            chart_box.line_chart(pd.DataFrame({"Loss": losses}))

        with st.spinner("正在训练字符级 RNN..."):
            model, stoi, itos, losses = train_char_rnn(corpus, hidden_size, epochs, learning_rate, update_training_view)
        st.session_state["char_rnn"] = (model, stoi, itos)
        st.session_state["char_rnn_losses"] = losses
        st.success("训练完成。")

    if "char_rnn_losses" in st.session_state:
        st.line_chart(pd.DataFrame({"Loss": st.session_state["char_rnn_losses"]}))

    if "char_rnn" in st.session_state:
        seed = st.text_input("Seed 起始文本", value="hello")
        length = st.slider("生成长度", 30, 160, 80)
        model, stoi, itos = st.session_state["char_rnn"]
        generated = generate_text(model, stoi, itos, seed, length)
        st.markdown(f'<div class="generated-text">{html.escape(generated)}</div>', unsafe_allow_html=True)


def render_pretrained_tab() -> None:
    st.markdown("### 模块 3：预训练架构对比 Masked LM vs Causal LM")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### BERT Masked LM")
        masked = st.text_input("带 [MASK] 的句子", value="The man went to the [MASK] to buy some milk.")
        with st.spinner("BERT 正在预测 [MASK]..."):
            st.dataframe(predict_mask(masked), use_container_width=True, hide_index=True)
        st.caption("BERT 同时利用 [MASK] 左右两侧的上下文，因此体现的是双向建模。")
    with right:
        st.markdown("#### GPT-2 Causal LM")
        prompt = st.text_area("Prompt 前缀", value="Natural language processing is", height=82)
        if st.button("生成 GPT-2 续写", use_container_width=True):
            with st.spinner("GPT-2 正在自回归生成..."):
                generated = generate_gpt2(prompt)
            st.markdown(f'<div class="generated-text">{html.escape(generated)}</div>', unsafe_allow_html=True)
        st.caption("GPT-2 只能从左到右预测下一个 token，因此体现的是单向自回归生成。")


def render_ppl_tab() -> None:
    st.markdown("### 模块 4：语言模型评价 Perplexity")
    text = st.text_area(
        "每行输入一个测试句子",
        value="Natural language processing is a fascinating field.\nThe cat sat on the mat.\nbook the useful reads student language about.",
        height=140,
    )
    sentences = [line.strip() for line in text.splitlines() if line.strip()]
    rows = []
    with st.spinner("正在使用 GPT-2 计算困惑度..."):
        for sentence in sentences:
            loss, ppl = sentence_perplexity(sentence)
            rows.append(
                {
                    "句子": sentence,
                    "Cross-Entropy Loss": loss,
                    "PPL = exp(Loss)": ppl,
                    "Token 数": max(len(tokenize_words(sentence)), 1),
                }
            )
    st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True, hide_index=True)
    st.markdown(
        """
        <div class="lm-formula">
            PPL = exp(Cross-Entropy Loss)。PPL 不是概率，而是模型平均“不确定性”的指数化结果。<br/>
            对基础版 GPT-2 来说，通顺短句得到几十到一百左右并不反常；真正重要的是同一模型下，
            合理句子的 PPL 应显著低于乱码句子。
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_language_model_app() -> None:
    render_lm_intro()
    tab1, tab2, tab3, tab4 = st.tabs(["n-gram 与平滑", "RNN 自训练", "BERT vs GPT-2", "困惑度 PPL"])
    with tab1:
        try:
            render_ngram_tab()
        except Exception as exc:
            st.error(f"n-gram 模块运行失败：{exc}")
    with tab2:
        try:
            render_rnn_tab()
        except Exception as exc:
            st.error(f"RNN 模块运行失败：{exc}")
    with tab3:
        try:
            render_pretrained_tab()
        except Exception as exc:
            st.error(f"预训练模型模块运行失败：{exc}")
    with tab4:
        try:
            render_ppl_tab()
        except Exception as exc:
            st.error(f"PPL 模块运行失败：{exc}")
