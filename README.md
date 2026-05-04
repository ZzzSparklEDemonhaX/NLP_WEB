# NLP_WEB

一个面向自然语言处理课程实验、课堂展示与项目答辩的综合型 Streamlit Web 平台。

本项目不是把若干零散算法简单堆在一起，而是尝试按照 NLP 的知识学习路径，把词法分析、句法分析、语义表示、深层语义、篇章分析、语言模型、信息抽取、机器翻译与情感分析组织成一个统一入口、统一风格、统一交互逻辑的九模块课程系统。

GitHub Repository:
[https://github.com/dimenghan2tech/NLP_WEB.git](https://github.com/dimenghan2tech/NLP_WEB.git)

## 1. Why This Project

在 NLP 课程学习中，常见问题不是“没有模型”，而是：

- 理论很多，但难以通过一个完整系统把知识串起来
- 单个算法能运行，但页面分散、展示不统一，不适合课堂汇报
- 预训练模型、规则方法、统计方法之间缺少直观对比
- 许多课程实验只停留在 notebook，缺少可交互、可演示、可部署的 Web 成果

这个项目的目标，就是把这些问题统一解决掉：

- 把九类 NLP 实验放进一个完整的 Web 系统
- 让每个模块不仅“能跑”，还“能讲”
- 让算法结果、原理说明、对比分析和可视化放在同一页面中
- 让本地展示、GitHub 托管、Streamlit 部署形成完整作品链路

## 2. What This Project Does

本项目构建了一个基于 Streamlit 的 NLP 教学展示平台，包含 9 个子应用：

1. 词法分析应用
2. 句法分析应用
3. 语义分析空间
4. 深层语义分析平台
5. 篇章分析综合平台
6. 语言模型训练与对比分析平台
7. 信息抽取实验平台
8. 机器翻译对比与评测应用
9. 情感分析可视化应用

系统首页不是简单的功能列表，而是按照 NLP 的知识链路组织成四层结构：

- 基础结构层：词法分析、句法分析
- 语义理解层：语义表示、深层语义、篇章分析
- 生成建模层：语言模型、机器翻译
- 知识应用层：信息抽取、情感分析

这使得项目更适合教学演示和答辩展示，而不是仅仅作为一个工具集合。

## 3. Problems Solved

这个项目重点解决了下面几类实际问题：

### 3.1 课程展示碎片化

将九个独立实验整合进同一个入口，避免“九个脚本、九个页面、九种风格”的割裂感。

### 3.2 理论与结果脱节

每个模块不仅展示结果，还补充：

- 算法逻辑
- 伪代码 / 公式
- baseline 对比
- 可视化解释
- 适用场景说明

这样页面不仅能操作，也能直接用于课堂讲解。

### 3.3 传统方法与现代模型缺少对照

项目中多处采用“规则 / 统计 / 神经网络 / 预训练模型”并排对照，例如：

- 中文分词算法对比
- Lesk 与上下文向量的词义消歧对比
- EDU 规则切分 vs NeuralEDUSeg 真实标注
- n-gram vs RNN vs BERT / GPT-2
- 规则直译 vs 神经机器翻译

这让课程中“模型演进”的主线能够被直观展示出来。

### 3.4 结果不可视、不可讲

项目对多个模块做了可视化增强，例如：

- 分词泡泡高亮
- 词性标签高亮
- 依存句法图
- 成分句法树
- LSA 二维散点图
- 情感仪表盘
- 知识图谱关系网络
- 指代消解高亮

这些可视化让抽象的 NLP 结果更适合课堂展示和答辩演示。

### 3.5 从本地实验到可发布项目的落差

项目补齐了：

- 项目结构组织
- requirements 依赖说明
- GitHub README
- 本地运行说明
- Streamlit 部署说明

因此它不仅是实验代码，也可以作为一个完整的 GitHub 项目被展示。

## 4. Core Features By Module

### 应用 1：词法分析应用

- 中文长文本输入
- 文本规范化
- 多种分词方式展示
- 词频统计与图表
- 词性标注与同类高亮
- 分词 / 词性标注方法介绍与对比

### 应用 2：句法分析应用

- 依存句法图渲染
- 成分句法树展示
- 英文与中文句法测试
- 结构歧义观察
- 核心论元提取
- 依存结构 vs 成分结构知识对比

### 应用 3：语义分析空间

- TF-IDF 关键词抽取
- LSA 降维
- Word2Vec 实时训练
- CBOW / Skip-Gram 对比
- GloVe 词类比
- FastText OOV 测试
- 简化版 Sent2Vec 句向量相似度

### 应用 4：深层语义分析平台

- WSD 词义消歧
- 两个不同语境下目标词词义并行分析
- Lesk 传统方法
- 基于上下文向量的动态语义区分
- 语义角色标注 SRL
- 谓词-论元结构抽取

### 应用 5：篇章分析综合平台

- EDU 话语分割
- 规则基线切分
- NeuralEDUSeg 真实标注对照
- 边界词与 EDU 末词高亮
- 浅层篇章关系提取
- 指代消解可视化

### 应用 6：语言模型训练与对比分析平台

- n-gram 语言模型
- Add-one 平滑对比
- 句子联合概率展示
- RNN 字符级训练
- GPT-2 续写
- BERT Mask 预测
- 困惑度 PPL 计算

### 应用 7：信息抽取实验平台

- NER 命名实体识别
- BIO 底层标注查看
- 关系抽取
- 结构化表格展示
- 知识图谱交互可视化

### 应用 8：机器翻译对比与评测应用

- 神经机器翻译
- 规则词典直译
- 神经翻译与直译对照
- BLEU 自动评测
- 习语翻译测试

### 应用 9：情感分析可视化应用

- 单句情感分类
- 情感置信度展示
- 显式情感与隐式情感对比
- 批量评论分析
- 舆情统计图表

## 5. Engineering Highlights

除了算法实验本身，这个项目还做了一些偏工程化的整合工作：

- 使用统一的模块注册机制管理九个应用
- 使用统一的首页、侧边栏和卡片式 UI
- 为不同模块设计了相对一致的视觉语言
- 支持本地调试与课堂展示
- 对部分模型和资源准备了缓存 / 回退逻辑
- 尽量把“算法结果 + 原理说明 + 页面交互”放在同一系统中

## 6. Tech Stack

### Web / Visualization

- Streamlit
- Plotly
- HTML / CSS
- NetworkX

### NLP / Machine Learning

- pandas
- numpy
- scikit-learn
- scipy
- nltk
- jieba
- opencc-python-reimplemented
- spaCy
- gensim
- torch
- transformers
- sentencepiece
- protobuf
- fastcoref
- benepar

## 7. Recommended Environment

- Python `3.11`
- 建议使用统一虚拟环境，例如 `.venv` 或 `cp311`

## 8. Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 9. Run Locally

```powershell
streamlit run app.py --server.port 8502
```

启动后访问：

- `http://localhost:8502`

## 10. Project Structure

```text
NLP_WEB/
├─ app.py
├─ requirements.txt
├─ README.md
├─ .gitignore
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

## 11. Deployment To Streamlit Community Cloud

把项目推到 GitHub 之后，可以按下面步骤部署：

1. 打开 [https://share.streamlit.io/](https://share.streamlit.io/)
2. 点击 `New app`
3. 选择仓库 `dimenghan2tech/NLP_WEB`
4. Branch 选择 `main`
5. Main file path 填写 `app.py`
6. 点击 `Deploy`

## 12. Git Commands

如果你想手动上传当前项目到 GitHub，可以使用下面这组命令：

```powershell
cd D:\desk\自然语言处理\NLP_WEB
git init
git branch -M main
git add .
git commit -m "feat: initial NLP course web platform"
git remote add origin https://github.com/dimenghan2tech/NLP_WEB.git
git push -u origin main
```

如果远程仓库已经设置过 `origin`，则改用：

```powershell
git remote set-url origin https://github.com/dimenghan2tech/NLP_WEB.git
git push -u origin main
```

## 13. Current Value Of The Project

这个项目适合以下用途：

- NLP 课程实验提交
- 课堂展示 / 结课答辩
- GitHub 项目作品集
- Streamlit Web 演示
- 个人 NLP 学习路径总结

## 14. Limitations

当前项目仍然有一些可继续优化的地方：

- 部分模块依赖较重，首次加载时间可能较长
- 云端部署时，大模型下载和运行资源可能受限
- 某些高级任务仍属于近似实现，而非完整工业级系统
- 中英文模型资源在不同环境下可能存在兼容性差异

## 15. Author

- GitHub: [dimenghan2tech](https://github.com/dimenghan2tech)

## 16. License

本项目主要用于课程学习、实验展示与个人项目演示。
