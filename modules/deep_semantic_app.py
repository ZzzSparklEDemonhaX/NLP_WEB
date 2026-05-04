from __future__ import annotations

import html
import re
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
import spacy
import streamlit as st
import torch
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from spacy import displacy
from transformers import AutoModel, AutoTokenizer


DEFAULT_WSD_SENTENCE_1 = "I went to the bank to deposit my money."
DEFAULT_WSD_SENTENCE_2 = "I sat by the river bank."
DEFAULT_TARGET_WORD = "bank"
DEFAULT_SRL_SENTENCE = "Apple is manufacturing new smartphones in China this year."

CORE_SRL_COLUMNS = ["A0 施事者", "Predicate 谓词", "A1 受事者", "AM-LOC 地点", "AM-TMP 时间"]

SYNSET_TRANSLATIONS = {
    "depository_financial_institution.n.01": "金融机构；接受存款并提供贷款等金融服务的银行。",
    "bank.n.01": "河岸；河流、湖泊等水体旁的斜坡或岸边土地。",
    "bank.n.09": "金融机构；银行。",
    "savings_bank.n.02": "储蓄银行。",
    "bank.v.01": "把钱存入银行。",
    "spring.n.01": "春天；万物生长的季节。",
    "spring.n.02": "弹簧；受压或拉伸后会恢复形状的金属弹性装置。",
    "spring.n.03": "泉水；地下水自然流出的水源。",
    "spring.n.04": "泉眼；水流涌出的地点。",
    "give.n.01": "弹性；被拉伸后恢复原状的能力。",
    "leap.n.01": "跳跃；向上或向前的轻快运动。",
    "jump.v.01": "跳跃；向前跳动。",
    "bounce.v.01": "弹回；受到撞击后反弹。",
    "spring.v.04": "突然出现或发展。",
    "spring.v.05": "突然提出、透露或释放。",
    "right.n.01": "权利；法律、传统或自然赋予个人或群体应得之物。",
    "right.n.02": "右边；右侧方向或位置。",
    "right.n.07": "正义；符合公平原则的事物。",
    "right.n.08": "权益；法律或习俗拥有的无形利益。",
    "correct.a.01": "正确的；没有错误，符合事实或真理。",
    "right.a.01": "右边的；位于或指向身体右侧的。",
    "right.a.04": "正当的；符合法律、道德或正义的。",
    "right.a.05": "判断正确的；观点或判断是对的。",
}

SENSE_DISPLAY_NAMES = {
    "bank.n.01": "bank｜名词｜河岸",
    "bank.n.09": "bank｜名词｜银行 / 金融机构",
    "depository_financial_institution.n.01": "bank｜名词｜银行 / 金融机构",
    "savings_bank.n.02": "bank｜名词｜储蓄银行",
    "bank.v.01": "bank｜动词｜存入银行",
    "spring.n.01": "spring｜名词｜春天 / 春季",
    "spring.n.02": "spring｜名词｜弹簧",
    "spring.n.03": "spring｜名词｜泉水",
    "spring.n.04": "spring｜名词｜泉眼",
    "give.n.01": "spring｜名词｜弹性",
    "leap.n.01": "spring｜名词｜跳跃",
    "jump.v.01": "spring｜动词｜跳跃",
    "bounce.v.01": "spring｜动词｜弹回 / 反弹",
    "spring.v.04": "spring｜动词｜突然出现",
    "spring.v.05": "spring｜动词｜突然提出 / 释放",
    "right.n.01": "right｜名词｜权利",
    "right.n.02": "right｜名词｜右边 / 右侧",
    "right.n.07": "right｜名词｜正义 / 正当",
    "right.n.08": "right｜名词｜权益",
    "correct.a.01": "right｜形容词｜正确的",
    "right.a.01": "right｜形容词｜右侧的",
    "right.a.04": "right｜形容词｜正当的",
    "right.a.05": "right｜形容词｜判断正确的",
}

POS_DISPLAY = {
    "n": "名词",
    "v": "动词",
    "a": "形容词",
    "s": "形容词",
    "r": "副词",
}

SPACY_TO_WORDNET_POS = {
    "NOUN": wn.NOUN,
    "PROPN": wn.NOUN,
    "VERB": wn.VERB,
    "AUX": wn.VERB,
    "ADJ": wn.ADJ,
    "ADV": wn.ADV,
}

ONLINE_POS_MAP = {
    "n": "noun",
    "v": "verb",
    "a": "adjective",
    "s": "adjective",
    "r": "adverb",
}

WSD_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "being", "been", "to", "of",
    "in", "on", "at", "for", "from", "with", "and", "or", "but", "that", "which",
    "who", "whom", "whose", "this", "these", "those", "it", "its", "as", "by",
    "has", "have", "had", "do", "does", "did", "not", "no", "very", "some",
}


@dataclass(frozen=True)
class BertResource:
    tokenizer: object | None
    model: object | None
    status: str


def run_module_command(args: list[str]) -> tuple[bool, str]:
    """执行资源下载命令，并把输出压缩成页面可读的摘要。"""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:  # pragma: no cover - 用于网络/权限异常兜底。
        return False, str(exc)
    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
    return completed.returncode == 0, output[-1000:]


@st.cache_resource(show_spinner=False)
def ensure_wordnet() -> str:
    """确保 Lesk 算法需要的 WordNet 语料可用；缺失时尝试自动下载。"""
    try:
        wn.synsets("bank")
        return "WordNet 已可用。"
    except LookupError:
        import nltk

        ok_wordnet = nltk.download("wordnet", quiet=True)
        ok_omw = nltk.download("omw-1.4", quiet=True)
        if ok_wordnet:
            return "WordNet 缺失，已自动下载。"
        return f"WordNet 下载失败，Lesk 将使用内置 bank 示例兜底。omw-1.4={ok_omw}"


@st.cache_resource(show_spinner=False)
def load_bert() -> BertResource:
    """加载 bert-base-uncased，用于提取目标词上下文向量。"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
        model = AutoModel.from_pretrained("bert-base-uncased", local_files_only=True)
        model.eval()
        return BertResource(tokenizer, model, "已从本地缓存加载 bert-base-uncased。")
    except Exception as local_exc:
        try:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = AutoModel.from_pretrained("bert-base-uncased")
            model.eval()
            return BertResource(tokenizer, model, "本地缓存缺失，已在线加载 bert-base-uncased。")
        except Exception as online_exc:
            return BertResource(None, None, f"BERT 加载失败：{short_error(online_exc or local_exc)}")


@st.cache_resource(show_spinner=False)
def load_spacy_en() -> tuple[spacy.language.Language, str]:
    """加载 spaCy 英文模型；缺失时尝试自动下载。"""
    try:
        return spacy.load("en_core_web_sm"), "已加载 en_core_web_sm。"
    except OSError:
        ok, output = run_module_command(["spacy", "download", "en_core_web_sm"])
        if ok:
            return spacy.load("en_core_web_sm"), "en_core_web_sm 缺失，已自动下载并加载。"
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp, f"en_core_web_sm 下载失败，SRL 将降级：{short_message(output)}"


def short_error(exc: Exception) -> str:
    """把异常压缩成适合页面展示的一句话。"""
    return short_message(str(exc))


def short_message(message: str) -> str:
    message = message.replace("\n", " ").strip()
    if "WinError" in message or "Connection" in message:
        return "网络连接失败。"
    if len(message) > 140:
        return message[:140] + "..."
    return message or "未知错误"


def simple_tokens(text: str) -> list[str]:
    """Lesk 输入使用简单英文 token，避免依赖 punkt 额外资源。"""
    return re.findall(r"[A-Za-z']+", text.lower())


def target_word_pos(sentence: str, target_word: str) -> tuple[str | None, str]:
    """Infer the target word POS from the current sentence with spaCy."""
    try:
        nlp, _ = load_spacy_en()
        doc = nlp(sentence)
        for token in doc:
            if token.text.lower() == target_word.lower():
                return SPACY_TO_WORDNET_POS.get(token.pos_), f"{token.pos_}/{token.tag_}"
        return None, "未在句子中定位到目标词"
    except Exception:
        return None, "spaCy 词性判断失败"


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_online_dictionary(target_word: str) -> list[dict[str, str]]:
    """Query an online dictionary API for the target word, then normalize definitions."""
    try:
        session = requests.Session()
        session.trust_env = False
        response = session.get(
            f"https://api.dictionaryapi.dev/api/v2/entries/en/{target_word.lower()}",
            timeout=8,
        )
        response.raise_for_status()
        rows: list[dict[str, str]] = []
        for entry in response.json():
            for meaning in entry.get("meanings", []):
                part = meaning.get("partOfSpeech", "")
                for item in meaning.get("definitions", []):
                    definition = item.get("definition", "")
                    if definition:
                        rows.append(
                            {
                                "part_of_speech": part,
                                "definition": definition,
                                "example": item.get("example", ""),
                            }
                        )
        return rows
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=3600)
def translate_definition_online(text: str) -> str:
    """Translate an English definition online; fall back silently if the service is unavailable."""
    if not text:
        return ""
    try:
        session = requests.Session()
        session.trust_env = False
        response = session.get(
            "https://api.mymemory.translated.net/get",
            params={"q": text[:450], "langpair": "en|zh-CN"},
            timeout=8,
        )
        response.raise_for_status()
        translated = response.json().get("responseData", {}).get("translatedText", "")
        return translated if translated and translated.lower() != text.lower() else ""
    except Exception:
        return ""


def phrase_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = simple_tokens(text)
    return set(zip(*(tokens[index:] for index in range(n)))) if len(tokens) >= n else set()


def match_online_definition(target_word: str, wn_pos: str | None, definition: str, sentence: str) -> tuple[str, str]:
    """Pick an online dictionary definition that matches the inferred POS and WordNet gloss."""
    online_rows = fetch_online_dictionary(target_word)
    if not online_rows:
        return "", "在线词典暂不可用，已使用本地 WordNet。"
    preferred_pos = ONLINE_POS_MAP.get(wn_pos or "")
    candidates = [
        row for row in online_rows if not preferred_pos or row["part_of_speech"].lower() == preferred_pos
    ] or online_rows
    definition_tokens = {token for token in simple_tokens(definition) if token not in WSD_STOPWORDS}
    context_tokens = {
        token for token in simple_tokens(sentence)
        if token not in WSD_STOPWORDS and token != target_word.lower()
    }
    target_to_pattern = re.search(rf"\b{re.escape(target_word.lower())}\s+to\s+(?!left\b|right\b)\w+", sentence.lower()) is not None

    def online_score(row: dict[str, str]) -> int:
        online_text = f"{row['definition']} {row.get('example', '')}"
        online_tokens = {token for token in simple_tokens(online_text) if token not in WSD_STOPWORDS}
        return (
            len(definition_tokens & online_tokens) * 2
            + len(context_tokens & online_tokens)
            + (10 if target_to_pattern and re.search(rf"\b{re.escape(target_word.lower())}\s+to\s+(?!left\b|right\b)\w+", online_text.lower()) else 0)
            + (2 if row["part_of_speech"].lower() == preferred_pos else 0)
        )

    best = max(candidates, key=online_score)
    if online_score(best) <= 2:
        return "", "在线词典未找到足够匹配当前语境的释义，已使用本地 WordNet。"
    source = f"在线词典 DictionaryAPI：{best['part_of_speech']}"
    if best.get("example"):
        source += f"；例句：{best['example']}"
    return best["definition"], source


def fallback_bank_sense(sentence: str) -> tuple[str, str, str]:
    """WordNet 不可用时的课堂兜底，只覆盖 bank 金融/河岸示例。"""
    lower = sentence.lower()
    if any(clue in lower for clue in ["money", "deposit", "loan", "account", "cash"]):
        return "bank.n.09", "a financial institution that accepts deposits and channels the money into lending activities", "内置兜底"
    if any(clue in lower for clue in ["river", "water", "shore", "stream"]):
        return "bank.n.01", "sloping land beside a body of water", "内置兜底"
    return "unknown", "未能根据内置规则判断词义。", "内置兜底"


def lesk_overlap_score(sentence_tokens: list[str], synset) -> int:
    """Score a candidate sense by overlap with its definition, examples, lemmas and hypernyms."""
    context = set(sentence_tokens)
    gloss_text = " ".join(
        [
            synset.definition(),
            " ".join(synset.examples()),
            " ".join(synset.lemma_names()),
            " ".join(lemma for hyper in synset.hypernyms() for lemma in hyper.lemma_names()),
        ]
    )
    return len(context & set(simple_tokens(gloss_text)))


def contextual_gloss_scores(sentence: str, target_word: str, candidates: list) -> dict[str, float]:
    """Rank candidate senses by comparing BERT target context with BERT gloss context."""
    try:
        resource = load_bert()
        sentence_vector, _ = bert_context_embedding(sentence, target_word, resource)
        if sentence_vector is None:
            return {}
        scores: dict[str, float] = {}
        for synset in candidates:
            gloss = f"{target_word} means {synset.definition()}. " + " ".join(synset.examples()[:1])
            gloss_vector, _ = bert_context_embedding(gloss, target_word, resource)
            score = cosine(sentence_vector, gloss_vector)
            if score is not None:
                # Prefer senses whose canonical lemma is the requested target word, not only a synonym.
                exact_lemma_bonus = 0.05 if synset.lemma_names() and synset.lemma_names()[0].lower() == target_word.lower() else 0.0
                scores[synset.name()] = score + exact_lemma_bonus + syntax_sense_bonus(sentence, target_word, synset)
        return scores
    except Exception:
        return {}


def syntax_sense_bonus(sentence: str, target_word: str, synset) -> float:
    """Small generic syntax bonus for constructions that are hard for gloss overlap alone."""
    lower = sentence.lower()
    definition = synset.definition().lower()
    examples = " ".join(synset.examples()).lower()
    if re.search(rf"\b{re.escape(target_word.lower())}\s+to\s+\w+", lower):
        if any(clue in f"{definition} {examples}" for clue in ["law", "custom", "due", "entitled", "entitlement", "rights"]):
            return 0.08
    return 0.0


def choose_target_synset(sentence: str, target_word: str):
    """Choose only from dynamic POS-matched WordNet senses of the requested target word."""
    inferred_pos, pos_note = target_word_pos(sentence, target_word)
    all_candidates = wn.synsets(target_word)
    candidates = [synset for synset in all_candidates if not inferred_pos or synset.pos() == inferred_pos]
    candidates = candidates or all_candidates
    if not candidates:
        return None, "WordNet 未找到该目标词的候选词义"
    bert_scores = contextual_gloss_scores(sentence, target_word, candidates[:16])
    if bert_scores:
        best = max(candidates[:16], key=lambda synset: bert_scores.get(synset.name(), -1.0))
        return best, f"spaCy 动态词性={pos_note}；BERT 上下文-释义相似度排序"
    tokens = [token for token in simple_tokens(sentence) if token != target_word.lower()]
    scored = sorted(
        ((lesk_overlap_score(tokens, synset), -index, synset) for index, synset in enumerate(candidates)),
        reverse=True,
    )
    best = scored[0][2]
    try:
        nltk_synset = lesk(simple_tokens(sentence), target_word.lower(), pos=inferred_pos)
        if nltk_synset in candidates and scored[0][0] == 0:
            best = nltk_synset
    except Exception:
        pass
    return best, f"spaCy 动态词性={pos_note}；目标词候选词义 Lesk overlap"


def display_synset_name(target_word: str, synset_name: str) -> str:
    """Render the sense around the user's target word instead of WordNet's canonical lemma."""
    if synset_name in SENSE_DISPLAY_NAMES:
        return SENSE_DISPLAY_NAMES[synset_name]
    try:
        synset = wn.synset(synset_name)
        pos = POS_DISPLAY.get(synset.pos(), synset.pos())
        return f"{target_word}｜{pos}｜{synset.definition()[:28]}"
    except Exception:
        return synset_name


def run_lesk(sentence: str, target_word: str) -> tuple[str, str, str]:
    """运行 Lesk 算法，返回 synset、definition 和状态。"""
    status = ensure_wordnet()
    try:
        synset, method = choose_target_synset(sentence, target_word)
        if synset is None:
            return "None", "Lesk 未找到合适词义。", status
        return synset.name(), synset.definition(), f"{status} 当前选择方法：{method}。"
    except LookupError:
        return fallback_bank_sense(sentence)


def translate_synset(synset_name: str, definition: str) -> str:
    """为 WordNet 词义提供中文说明；常见课堂词义用人工翻译，其他词义保留英文释义。"""
    if synset_name in SYNSET_TRANSLATIONS:
        return SYNSET_TRANSLATIONS[synset_name]
    if synset_name.startswith("bank.") and "financial" in definition.lower():
        return "金融含义：银行或金融机构。"
    if synset_name.startswith("bank.") and any(word in definition.lower() for word in ["river", "water", "slope"]):
        return "地理含义：河岸或水边斜坡。"
    return f"中文释义待补充，可参考英文定义：{definition}"


def dictionary_lookup(sentence: str, target_word: str) -> dict[str, str]:
    """针对具体句子中的目标词，输出 Lesk 选中的 WordNet 词义和中文解释。"""
    synset_name, definition, _ = run_lesk(sentence, target_word)
    wn_pos, pos_note = target_word_pos(sentence, target_word)
    online_definition, online_source = match_online_definition(target_word, wn_pos, definition, sentence)
    translation_source = online_definition or definition
    zh_translation = translate_definition_online(translation_source) or translate_synset(synset_name, definition)
    examples = ""
    try:
        synsets = wn.synsets(target_word)
        synset = next((item for item in synsets if item.name() == synset_name), None)
        if synset is not None:
            examples = "；".join(synset.examples()[:2])
    except LookupError:
        examples = ""
    display_name = display_synset_name(target_word, synset_name)
    return {
        "句子": sentence,
        "目标词": target_word,
        "WordNet Synset": display_name,
        "WordNet ID": synset_name,
        "动态词性": pos_note,
        "英文定义": definition,
        "在线词典定义": online_definition or "-",
        "在线来源": online_source,
        "中文释义": zh_translation,
        "例句": examples or "-",
    }


def find_target_piece_indices(offsets: list[tuple[int, int]], sentence: str, target_word: str) -> list[int]:
    """根据 tokenizer offset 找到目标词对应的 wordpiece 下标。"""
    match = re.search(rf"\b{re.escape(target_word)}\b", sentence, flags=re.IGNORECASE)
    if match is None:
        return []
    start, end = match.span()
    indices = []
    for index, (piece_start, piece_end) in enumerate(offsets):
        if piece_start == piece_end == 0:
            continue
        if piece_start < end and piece_end > start:
            indices.append(index)
    return indices


def bert_context_embedding(sentence: str, target_word: str, resource: BertResource) -> tuple[np.ndarray | None, str]:
    """提取目标词的 BERT 上下文向量；若被拆成多个 wordpiece，则取平均。"""
    if resource.tokenizer is None or resource.model is None:
        return None, resource.status

    encoded = resource.tokenizer(sentence, return_offsets_mapping=True, return_tensors="pt", truncation=True)
    offsets = [(int(start), int(end)) for start, end in encoded.pop("offset_mapping")[0].tolist()]
    indices = find_target_piece_indices(offsets, sentence, target_word)
    if not indices:
        return None, f"句子中没有找到目标词 `{target_word}`。"

    with torch.no_grad():
        outputs = resource.model(**encoded)
    hidden = outputs.last_hidden_state[0]
    vector = hidden[indices].mean(dim=0).detach().cpu().numpy()
    return vector, f"目标词由 {len(indices)} 个 wordpiece 表示，已取平均向量。"


def cosine(vec_a: np.ndarray | None, vec_b: np.ndarray | None) -> float | None:
    """计算两个上下文向量的余弦相似度。"""
    if vec_a is None or vec_b is None:
        return None
    denominator = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denominator == 0:
        return None
    return float(np.dot(vec_a, vec_b) / denominator)


def displacy_svg(doc: spacy.tokens.Doc) -> str | None:
    """渲染 spaCy 依存句法 SVG。"""
    if not any(token.dep_ for token in doc):
        return None
    options = {
        "compact": True,
        "bg": "transparent",
        "color": "#18212f",
        "distance": 96,
        "arrow_stroke": 2,
    }
    return displacy.render(doc, style="dep", options=options, jupyter=False)


def subtree_text(token: spacy.tokens.Token) -> str:
    """返回依存子树对应的连续文本片段。"""
    tokens = sorted(token.subtree, key=lambda item: item.i)
    return " ".join(item.text for item in tokens)


def extract_srl_roles(sentence: str) -> tuple[pd.DataFrame, spacy.tokens.Doc, str]:
    """使用 spaCy 依存关系近似抽取谓词-论元结构。"""
    nlp, status = load_spacy_en()
    doc = nlp(sentence)
    rows = []

    predicates = [token for token in doc if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}]
    predicates.extend(token for token in doc if token.pos_ == "VERB" and token not in predicates)
    predicates = predicates[:3]

    for predicate in predicates:
        role_map = {
            "A0 施事者": "",
            "Predicate 谓词": predicate.text,
            "A1 受事者": "",
            "AM-LOC 地点": "",
            "AM-TMP 时间": "",
        }

        for child in predicate.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                role_map["A0 施事者"] = subtree_text(child)
            elif child.dep_ in {"dobj", "obj", "attr", "oprd"}:
                role_map["A1 受事者"] = subtree_text(child)
            elif child.dep_ == "prep":
                pobj = next((grand for grand in child.children if grand.dep_ == "pobj"), None)
                phrase = subtree_text(child)
                if child.text.lower() in {"in", "at", "on", "near", "from", "inside", "outside"} and pobj is not None:
                    role_map["AM-LOC 地点"] = phrase
                elif child.text.lower() in {"during", "before", "after", "since", "until"}:
                    role_map["AM-TMP 时间"] = phrase

        # 命名实体能弥补部分依存规则抓不到的时间/地点修饰语。
        loc_entities = [ent.text for ent in doc.ents if ent.label_ in {"GPE", "LOC", "FAC"}]
        time_entities = [ent.text for ent in doc.ents if ent.label_ in {"DATE", "TIME"}]
        if loc_entities and not role_map["AM-LOC 地点"]:
            role_map["AM-LOC 地点"] = ", ".join(loc_entities)
        if time_entities and not role_map["AM-TMP 时间"]:
            role_map["AM-TMP 时间"] = ", ".join(time_entities)

        rows.append(role_map)

    if not rows:
        rows.append({column: "" for column in CORE_SRL_COLUMNS})
    return pd.DataFrame(rows, columns=CORE_SRL_COLUMNS), doc, status


def render_deep_semantic_intro() -> None:
    st.markdown(
        """
        <section class="module-hero deep-semantic-hero" style="--accent:#ef4444">
            <p class="eyebrow">APP 04 · DEEP SEMANTIC ANALYSIS</p>
            <h1>深层语义分析平台</h1>
            <p>围绕词义消歧与语义角色标注，比较词典释义匹配和上下文向量表示，并抽取“谁对谁在何处何时做了什么”。</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_wsd_tab() -> None:
    st.markdown("### 模块 1：词义消歧 WSD 对比测试")
    st.markdown(
        """
        <div class="deep-note">
            Lesk 使用 WordNet 释义重合度选择词义；BERT 使用上下文动态向量表示同一个词在不同句子中的语义差异。
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        sentence_1 = st.text_area("句子 1", value=DEFAULT_WSD_SENTENCE_1, height=96)
    with col_b:
        sentence_2 = st.text_area("句子 2", value=DEFAULT_WSD_SENTENCE_2, height=96)
    target_word = st.text_input("目标多义词", value=DEFAULT_TARGET_WORD).strip().lower()

    st.markdown("#### 两个语境中的词典查找结果")
    lookup_1 = dictionary_lookup(sentence_1, target_word)
    lookup_2 = dictionary_lookup(sentence_2, target_word)
    _, _, lesk_status = run_lesk(sentence_1, target_word)
    st.caption(lesk_status)
    dict_col_1, dict_col_2 = st.columns(2, gap="large")
    with dict_col_1:
        st.markdown(
            f"""
            <div class="sense-card">
                <strong>句子 1：{html.escape(lookup_1["WordNet Synset"])}</strong>
                <p><b>动态词性：</b>{html.escape(lookup_1["动态词性"])}</p>
                <p><b>在线词典：</b>{html.escape(lookup_1["在线词典定义"])}</p>
                <p><b>WordNet：</b>{html.escape(lookup_1["英文定义"])}</p>
                <p><b>中文释义：</b>{html.escape(lookup_1["中文释义"])}</p>
                <p><b>来源：</b>{html.escape(lookup_1["在线来源"])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with dict_col_2:
        st.markdown(
            f"""
            <div class="sense-card">
                <strong>句子 2：{html.escape(lookup_2["WordNet Synset"])}</strong>
                <p><b>动态词性：</b>{html.escape(lookup_2["动态词性"])}</p>
                <p><b>在线词典：</b>{html.escape(lookup_2["在线词典定义"])}</p>
                <p><b>WordNet：</b>{html.escape(lookup_2["英文定义"])}</p>
                <p><b>中文释义：</b>{html.escape(lookup_2["中文释义"])}</p>
                <p><b>来源：</b>{html.escape(lookup_2["在线来源"])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    bert = load_bert()
    vec_1, note_1 = bert_context_embedding(sentence_1, target_word, bert)
    vec_2, note_2 = bert_context_embedding(sentence_2, target_word, bert)
    score = cosine(vec_1, vec_2)

    st.markdown("#### BERT 上下文向量对比")
    st.caption(f"{bert.status} {note_1} {note_2}")
    if score is None:
        st.warning("无法计算两个上下文向量的相似度，请检查目标词是否同时出现在两个句子中。")
    else:
        st.metric("两个 bank 上下文向量的余弦相似度", f"{score:.4f}")
        st.markdown(
            '<div class="deep-formula">cos(v₁,v₂)=v₁·v₂/(||v₁||||v₂||)。不同语境下相似度下降，说明 BERT 表示是动态上下文相关的。</div>',
            unsafe_allow_html=True,
        )


def render_srl_tab() -> None:
    st.markdown("### 模块 2：语义角色标注 SRL 提取与可视化")
    sentence = st.text_area(
        "输入英文句子",
        value=DEFAULT_SRL_SENTENCE,
        height=92,
        help="也可尝试：Chengfei Company is manufacturing civil aircrafts.",
    )

    role_df, doc, status = extract_srl_roles(sentence)
    st.caption(status)
    st.markdown("#### 谓词-论元结构")
    st.dataframe(role_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="deep-role-grid">
            <article><span>A0</span><p>施事者 Agent，通常由 nsubj 主语映射而来。</p></article>
            <article><span>A1</span><p>受事者 Patient，通常由 dobj / obj 直接宾语映射而来。</p></article>
            <article><span>AM-LOC</span><p>地点修饰语，可由地点介词短语或 GPE/LOC 实体识别。</p></article>
            <article><span>AM-TMP</span><p>时间修饰语，可由 DATE/TIME 实体或时间介词短语识别。</p></article>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 依存图辅助验证")
    svg = displacy_svg(doc)
    if svg:
        st.markdown(f'<div class="dependency-svg">{svg}</div>', unsafe_allow_html=True)
    else:
        st.warning("当前 spaCy 管线没有依存句法分析器，无法渲染依存图。")


def render_deep_semantic_app() -> None:
    render_deep_semantic_intro()
    tab_wsd, tab_srl = st.tabs(["词义消歧 WSD", "语义角色标注 SRL"])
    with tab_wsd:
        try:
            render_wsd_tab()
        except Exception as exc:
            st.error(f"WSD 模块运行失败：{short_error(exc)}")
    with tab_srl:
        try:
            render_srl_tab()
        except Exception as exc:
            st.error(f"SRL 模块运行失败：{short_error(exc)}")
