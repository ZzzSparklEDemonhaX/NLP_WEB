# NLP_WEB

一个面向自然语言处理课程展示的 Streamlit Web 项目。项目将 NLP 知识链路拆分为 9 个可交互应用，从词法、句法、语义、篇章，到语言模型、信息抽取、机器翻译与情感分析，适合课程实验、课堂展示与作品集发布。

## Project Overview

本项目以 `Streamlit` 作为统一前端框架，以“一个入口 + 多模块联动”的形式组织整个课程系统。首页不是简单的九宫格，而是按照 NLP 的知识逻辑组织为四层：

- 基础结构层：词法分析、句法分析
- 语义理解层：语义表示、深层语义分析、篇章分析
- 生成建模层：语言模型、机器翻译
- 知识应用层：信息抽取、情感分析

## Applications

### 1. 词法分析应用
- 文本规范化
- 中文分词结果展示
- 词频统计与可视化
- 词性标注与高亮
- 多种分词 / 词性标注方法对比

### 2. 句法分析应用
- 依存句法图展示
- 成分句法树展示
- 英文与中文句法输入测试
- 核心论元提取
- 依存结构与成分结构知识对比

### 3. 语义分析空间
- TF-IDF 关键词提取
- LSA 降维可视化
- Word2Vec 实时训练与相似词查询
- GloVe 词类比计算
- FastText OOV 测试
- 简化版 Sent2Vec 句向量相似度

### 4. 深层语义分析平台
- WSD 词义消歧
- Lesk 与上下文向量对比
- 两句语境下目标词的动态词义分析
- 语义角色标注 SRL
- 依存结构辅助语义理解

### 5. 篇章分析综合平台
- EDU 话语分割
- 规则基线切分 vs NeuralEDUSeg 真实标注
- 边界词与 EDU 末词高亮
- 浅层篇章关系提取
- 指代消解可视化

### 6. 语言模型训练与对比分析平台
- n-gram 语言模型
- Add-one 平滑对比
- 字符级 RNN 自训练与文本生成
- BERT Masked LM 与 GPT-2 Causal LM 对比
- 基于 GPT-2 的困惑度计算

### 7. 信息抽取实验平台
- 命名实体识别
- BIO 标注查看
- 关系抽取表格化展示
- 知识图谱交互可视化

### 8. 机器翻译对比与评测应用
- 神经机器翻译
- 规则直译 vs NMT 对比
- BLEU 自动评测
- 习语与复杂句测试

### 9. 情感分析可视化应用
- 单句情感分类
- 置信度仪表盘
- 显式 / 隐式情感对比
- 批量舆情数据统计与可视化

## Tech Stack

### Frontend / App
- Streamlit
- Plotly
- HTML / CSS

### NLP / ML
- pandas
- numpy
- scikit-learn
- scipy
- nltk
- jieba
- opencc-python-reimplemented
- spacy
- gensim
- torch
- transformers
- sentencepiece
- protobuf
- fastcoref
- benepar
- networkx

## Recommended Environment

- Python `3.11`
- 虚拟环境建议统一使用 `.venv` 或 `cp311`

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run Locally

```powershell
streamlit run app.py --server.port 8502
```

启动后访问：

- Local: `http://localhost:8502`

## Project Structure

```text
NLP_WEB/
├─ app.py
├─ requirements.txt
├─ README.md
├─ .streamlit/
│  └─ config.toml
├─ assets/
│  └─ styles.css
├─ data/
├─ logs/
└─ modules/
   ├─ registry.py
   ├─ ui.py
   ├─ lexical_app.py
   ├─ syntax_app.py
   ├─ semantic_app.py
   ├─ deep_semantic_app.py
   ├─ discourse_app.py
   ├─ language_model_app.py
   ├─ information_extraction_app.py
   ├─ machine_translation_app.py
   ├─ sentiment_app.py
   └─ __init__.py
```

## Deployment To Streamlit Community Cloud

如果你想把它部署到 Streamlit 云端，推荐按下面流程：

1. 将本项目推送到 GitHub 仓库。
2. 打开 [Streamlit Community Cloud](https://share.streamlit.io/)。
3. 选择 `New app`。
4. 选择你的仓库：`dimenghan2tech/NLP_WEB`
5. Branch 选择：`main`
6. Main file path 填写：`app.py`
7. 点击 `Deploy`

如果部署时遇到模型下载较慢、依赖较重、内存不足等问题，可以先在本地完成课堂展示，再针对云部署做轻量化裁剪。

## Git Commands

下面这组命令可以把当前项目初始化并推送到你的 GitHub 仓库：

```powershell
cd D:\desk\自然语言处理\NLP_WEB
git init
git branch -M main
git add .
git commit -m "feat: initial NLP course web platform"
git remote add origin https://github.com/dimenghan2tech/NLP_WEB.git
git push -u origin main
```

如果远程仓库已经存在且之前绑定过 origin，可以改用：

```powershell
git remote set-url origin https://github.com/dimenghan2tech/NLP_WEB.git
git push -u origin main
```

## Notes

- 部分模块依赖预训练模型，首次运行可能需要下载模型文件。
- 某些英文 / 中文分析模块依赖 spaCy 或 benepar 模型。
- 若网络不稳定，建议先在本地环境完成模型缓存。
- `logs/` 目录主要用于本地调试，不建议上传大量运行日志。

## Author

- GitHub: [dimenghan2tech](https://github.com/dimenghan2tech)

## License

仅用于课程学习、实验展示与个人项目演示。
