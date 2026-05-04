from dataclasses import dataclass
from typing import Callable

from modules import (
    deep_semantic_app,
    discourse_app,
    information_extraction_app,
    language_model_app,
    lexical_app,
    machine_translation_app,
    semantic_app,
    sentiment_app,
    syntax_app,
)


@dataclass(frozen=True)
class NlpApp:
    key: str
    title: str
    subtitle: str
    icon: str
    accent: str
    render: Callable[[], None]


APP_REGISTRY: list[NlpApp] = [
    NlpApp(
        key="lexical_lab",
        title="词法分析应用",
        subtitle="规范化、中文分词、词频统计、词性标注",
        icon="01",
        accent="#f59e0b",
        render=lexical_app.render_lexical_app,
    ),
    NlpApp(
        key="syntax_lab",
        title="句法分析应用",
        subtitle="依存句法图、成分句法树、结构歧义分析",
        icon="02",
        accent="#06b6d4",
        render=syntax_app.render_syntax_app,
    ),
    NlpApp(
        key="semantic_space_lab",
        title="语义分析空间",
        subtitle="TF-IDF、LSA、Word2Vec、GloVe、FastText",
        icon="03",
        accent="#22c55e",
        render=semantic_app.render_semantic_app,
    ),
    NlpApp(
        key="deep_semantic_lab",
        title="深层语义分析平台",
        subtitle="词义消歧、上下文向量、语义角色标注",
        icon="04",
        accent="#ef4444",
        render=deep_semantic_app.render_deep_semantic_app,
    ),
    NlpApp(
        key="discourse_lab",
        title="篇章分析综合平台",
        subtitle="EDU 切分、浅层篇章关系、指代消解",
        icon="05",
        accent="#8b5cf6",
        render=discourse_app.render_discourse_app,
    ),
    NlpApp(
        key="language_model_lab",
        title="语言模型训练与对比分析平台",
        subtitle="n-gram、RNN、BERT、GPT-2、困惑度",
        icon="06",
        accent="#14b8a6",
        render=language_model_app.render_language_model_app,
    ),
    NlpApp(
        key="ie_lab",
        title="信息抽取实验平台",
        subtitle="NER、BIO 标注、关系抽取、知识图谱",
        icon="07",
        accent="#3b82f6",
        render=information_extraction_app.render_information_extraction_app,
    ),
    NlpApp(
        key="mt_lab",
        title="机器翻译对比与评测应用",
        subtitle="NMT、规则直译、BLEU 自动评测",
        icon="08",
        accent="#f97316",
        render=machine_translation_app.render_machine_translation_app,
    ),
    NlpApp(
        key="sentiment_lab",
        title="情感分析可视化应用",
        subtitle="情感极性、显式与隐式情感、舆情仪表盘",
        icon="09",
        accent="#84cc16",
        render=sentiment_app.render_sentiment_app,
    ),
]
