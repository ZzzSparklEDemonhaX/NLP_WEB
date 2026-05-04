from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass

import pandas as pd
import spacy
import streamlit as st
import streamlit.components.v1 as components


DEFAULT_IE_TEXT = (
    "Steve Jobs founded Apple in California. Apple acquired Beats in 2014. "
    "Tim Cook leads Apple, and Apple is located in Cupertino."
)

ENTITY_COLORS = {
    "PERSON": "#bfdbfe",
    "ORG": "#bbf7d0",
    "GPE": "#fed7aa",
    "LOC": "#fde68a",
    "DATE": "#ddd6fe",
    "PRODUCT": "#fecdd3",
    "EVENT": "#c7d2fe",
    "MISC": "#e5e7eb",
}

GRAPH_COLORS = {
    "PERSON": "#2563eb",
    "ORG": "#16a34a",
    "GPE": "#f97316",
    "LOC": "#eab308",
    "DATE": "#8b5cf6",
    "PRODUCT": "#ef4444",
    "EVENT": "#6366f1",
    "MISC": "#64748b",
}

CHINESE_ENTITY_PATTERNS = {
    "PERSON": ["马云", "乔布斯", "雷军", "张一鸣", "任正非"],
    "ORG": ["阿里巴巴", "腾讯", "苹果", "小米", "字节跳动", "华为"],
    "GPE": ["北京", "上海", "杭州", "深圳", "中国", "美国", "加州"],
}


@dataclass(frozen=True)
class Entity:
    text: str
    label: str
    start: int
    end: int


@dataclass(frozen=True)
class Relation:
    subject: str
    predicate: str
    object: str


@st.cache_resource(show_spinner=False)
def load_spacy_model(language: str):
    """按语言加载 spaCy 模型。"""
    model_name = "zh_core_web_sm" if language == "zh" else "en_core_web_sm"
    try:
        return spacy.load(model_name), f"已加载 {model_name}。"
    except OSError:
        return spacy.blank(language), f"未找到 {model_name}，已退回规则补强模式。"


def contains_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def extract_entities(text: str) -> tuple[list[Entity], str]:
    """使用 spaCy NER + 规则补强抽取实体。"""
    language = "zh" if contains_chinese(text) else "en"
    nlp, status = load_spacy_model(language)
    doc = nlp(text)
    entities = [
        Entity(ent.text, normalize_label(ent.label_), ent.start_char, ent.end_char)
        for ent in doc.ents
        if ent.text.strip()
    ]

    if language == "zh":
        entities.extend(rule_chinese_entities(text))
    entities.extend(rule_english_entities(text, entities))
    return merge_entities(entities), status


def normalize_label(label: str) -> str:
    if label in {"PER", "PERSON"}:
        return "PERSON"
    if label in {"ORG", "ORGANIZATION"}:
        return "ORG"
    if label in {"GPE", "LOC", "LOCATION"}:
        return "GPE" if label == "GPE" else "LOC"
    if label in ENTITY_COLORS:
        return label
    return "MISC"


def rule_chinese_entities(text: str) -> list[Entity]:
    """中文 NER 规则补强，避免小模型识别不稳定影响课堂展示。"""
    results = []
    for label, names in CHINESE_ENTITY_PATTERNS.items():
        for name in names:
            for match in re.finditer(re.escape(name), text):
                results.append(Entity(name, label, match.start(), match.end()))
    return results


def rule_english_entities(text: str, existing: list[Entity]) -> list[Entity]:
    """英文规则补强：spaCy 偶尔漏掉产品或公司名时补上常见演示实体。"""
    known = {
        "Apple": "ORG",
        "Beats": "ORG",
        "OpenAI": "ORG",
        "Microsoft": "ORG",
        "Google": "ORG",
        "Cupertino": "GPE",
        "California": "GPE",
    }
    occupied = {(entity.start, entity.end) for entity in existing}
    results = []
    for name, label in known.items():
        for match in re.finditer(rf"\b{re.escape(name)}\b", text):
            if (match.start(), match.end()) not in occupied:
                results.append(Entity(name, label, match.start(), match.end()))
    return results


def merge_entities(entities: list[Entity]) -> list[Entity]:
    """合并重叠实体，优先保留更长跨度。"""
    sorted_entities = sorted(entities, key=lambda item: (item.start, -(item.end - item.start)))
    merged = []
    for entity in sorted_entities:
        if any(not (entity.end <= kept.start or entity.start >= kept.end) for kept in merged):
            continue
        merged.append(entity)
    return sorted(merged, key=lambda item: item.start)


def render_entities_html(text: str, entities: list[Entity]) -> str:
    """将实体在原文中高亮。"""
    parts = []
    cursor = 0
    for entity in entities:
        parts.append(html.escape(text[cursor : entity.start]))
        color = ENTITY_COLORS.get(entity.label, ENTITY_COLORS["MISC"])
        parts.append(
            f'<mark class="ie-entity" style="--entity-color:{color}">'
            f"{html.escape(entity.text)}<small>{entity.label}</small></mark>"
        )
        cursor = entity.end
    parts.append(html.escape(text[cursor:]))
    return '<div class="ie-text">' + "".join(parts) + "</div>"


def bio_tags(text: str, entities: list[Entity]) -> pd.DataFrame:
    """把实体跨度转换为简化 BIO 序列。"""
    tokens = [(match.group(0), match.start(), match.end()) for match in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE)]
    rows = []
    for token, start, end in tokens:
        tag = "O"
        for entity in entities:
            if start >= entity.start and end <= entity.end:
                prefix = "B" if start == entity.start else "I"
                tag = f"{prefix}-{entity.label}"
                break
        rows.append({"Token": token, "BIO Tag": tag})
    return pd.DataFrame(rows)


def extract_relations(text: str, entities: list[Entity]) -> list[Relation]:
    """基于触发词和实体顺序的轻量关系抽取。"""
    relations = []
    sentence_matches = list(re.finditer(r"[^.!?。！？]+[.!?。！？]?", text))
    for match in sentence_matches:
        sentence = match.group(0)
        sent_entities = [entity for entity in entities if entity.start >= match.start() and entity.end <= match.end()]
        if len(sent_entities) < 2:
            continue
        lower = sentence.lower()
        subject_entity = sent_entities[0]
        object_entity = sent_entities[1]
        if re.search(r"\bfounded|founded by|创立|创办|创建", lower):
            relations.append(Relation(subject_entity.text, "FOUNDER_OF", object_entity.text))
        if re.search(r"\bacquired|bought|收购", lower):
            relations.append(Relation(subject_entity.text, "ACQUIRED", object_entity.text))
        if re.search(r"\bleads|ceo|president|领导|担任", lower):
            org = next((entity for entity in sent_entities if entity.label == "ORG"), object_entity)
            person = next((entity for entity in sent_entities if entity.label == "PERSON"), subject_entity)
            relations.append(Relation(person.text, "LEADS", org.text))
        if re.search(r"\blocated|headquartered|based|位于|总部", lower):
            org = next((entity for entity in sent_entities if entity.label == "ORG"), subject_entity)
            place = next((entity for entity in reversed(sent_entities) if entity.label in {"GPE", "LOC"}), sent_entities[-1])
            relations.append(Relation(org.text, "LOCATED_IN", place.text))
        if re.search(r"\bworks at|joined|任职|加入", lower):
            relations.append(Relation(subject_entity.text, "WORKS_FOR", object_entity.text))

    return dedupe_relations(relations)


def dedupe_relations(relations: list[Relation]) -> list[Relation]:
    seen = set()
    unique = []
    for relation in relations:
        key = (relation.subject, relation.predicate, relation.object)
        if key not in seen:
            seen.add(key)
            unique.append(relation)
    return unique


def entities_dataframe(entities: list[Entity]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"实体": entity.text, "类型": entity.label, "起始": entity.start, "结束": entity.end} for entity in entities]
    )


def relations_dataframe(relations: list[Relation]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Subject 主体": rel.subject, "Predicate 关系": rel.predicate, "Object 客体": rel.object} for rel in relations]
    )


def render_graph(entities: list[Entity], relations: list[Relation]) -> None:
    """使用 vis-network 渲染可拖拽缩放知识图谱。"""
    unique_entities = {}
    for entity in entities:
        unique_entities.setdefault(entity.text, entity.label)

    nodes = [
        {
            "id": text,
            "label": text,
            "group": label,
            "color": GRAPH_COLORS.get(label, GRAPH_COLORS["MISC"]),
            "value": 18 if label in {"PERSON", "ORG"} else 12,
        }
        for text, label in unique_entities.items()
    ]
    edges = [
        {"from": relation.subject, "to": relation.object, "label": relation.predicate, "arrows": "to"}
        for relation in relations
        if relation.subject in unique_entities and relation.object in unique_entities
    ]

    graph_html = f"""
    <div id="ie-network" style="width:100%;height:500px;border-radius:22px;background:linear-gradient(135deg, rgba(255,255,255,0.96), rgba(239,246,255,0.88));"></div>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script>
      const renderGraph = () => {{
        const container = document.getElementById("ie-network");
        if (!container || typeof vis === "undefined") return;
        const nodes = new vis.DataSet({json.dumps(nodes, ensure_ascii=False)});
        const edges = new vis.DataSet({json.dumps(edges, ensure_ascii=False)});
        const data = {{ nodes, edges }};
        const options = {{
          autoResize: true,
          nodes: {{
            shape: "dot",
            font: {{ color: "#18212f", size: 16, face: "Arial" }},
            borderWidth: 2,
            shadow: true
          }},
          edges: {{
            color: "#64748b",
            font: {{ align: "middle", size: 13 }},
            smooth: {{ type: "dynamic" }},
            arrows: {{ to: {{ enabled: true, scaleFactor: 0.8 }} }}
          }},
          interaction: {{ hover: true, zoomView: true, dragNodes: true, navigationButtons: true }},
          physics: {{ stabilization: true, barnesHut: {{ gravitationalConstant: -2400, springLength: 130 }} }}
        }};
        new vis.Network(container, data, options);
      }};
      if (typeof vis === "undefined") {{
        window.addEventListener("load", renderGraph);
      }} else {{
        renderGraph();
      }}
    </script>
    """
    components.html(graph_html, height=520)


def render_ie_intro() -> None:
    st.markdown(
        """
        <section class="module-hero ie-hero" style="--accent:#3b82f6">
            <p class="eyebrow">APP 07 · INFORMATION EXTRACTION</p>
            <h1>信息抽取实验平台</h1>
            <p>从非结构化文本中抽取命名实体、BIO 底层标签、实体关系，并将线性文本转换为可交互知识图谱。</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_information_extraction_app() -> None:
    render_ie_intro()
    text = st.text_area("输入英文或中文语料", value=DEFAULT_IE_TEXT, height=150)
    show_bio = st.checkbox("查看底层 BIO 标注模式", value=False)

    entities, status = extract_entities(text)
    relations = extract_relations(text, entities)
    st.caption(status)

    tab_ner, tab_re, tab_kg = st.tabs(["NER & BIO", "关系抽取", "知识图谱"])
    with tab_ner:
        st.markdown("### 模块 1：命名实体识别与 BIO 标注")
        if show_bio:
            st.dataframe(bio_tags(text, entities), use_container_width=True, hide_index=True)
        else:
            st.markdown(render_entities_html(text, entities), unsafe_allow_html=True)
        st.markdown("#### 实体表")
        if entities:
            st.dataframe(entities_dataframe(entities), use_container_width=True, hide_index=True)
        else:
            st.info("没有识别到实体。")
        st.info("嵌套实体观察：University of California, Los Angeles 中 Los Angeles 本身也是地点，单层 BIO 很难同时表达外层组织和内层地点。")

    with tab_re:
        st.markdown("### 模块 2：实体关系抽取")
        if relations:
            st.dataframe(relations_dataframe(relations), use_container_width=True, hide_index=True)
        else:
            st.warning("没有抽取到关系。可尝试 founded / acquired / located in / leads 等触发词。")
        st.caption("关系抽取本质上是在实体节点之间预测语义边：Subject --Predicate--> Object。")

    with tab_kg:
        st.markdown("### 模块 3：知识图谱交互可视化")
        if entities and relations:
            render_graph(entities, relations)
        else:
            st.info("需要至少两个实体和一条关系才能生成图谱。")
