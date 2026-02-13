<h1 align="center">📄 Intelligent PDF Text Summarization System</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-%233776AB.svg?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/NLP-NLTK-%237B68EE.svg?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/TextRank-Graph%20Ranking-%23FF9800.svg?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PDF-Processing-%23E53935.svg?style=for-the-badge"/>
</p>

<hr/>

<h2>📌 Overview</h2>
<p>
This project presents an intelligent Natural Language Processing system designed to generate concise summaries
from PDF documents. The system extracts textual content, evaluates semantic relationships between sentences,
and ranks them using a graph-based TextRank algorithm to produce a meaningful condensed representation of the document.
</p>

---

<h2>🧠 Key Features</h2>

<ul>
<li>Automated PDF text extraction</li>
<li>Noise filtering and linguistic validation</li>
<li>Sentence similarity modeling using cosine distance</li>
<li>Graph-based ranking through PageRank</li>
<li>Domain-focused sentence prioritization</li>
<li>Context-aware summary enhancement</li>
<li>Command-line execution interface</li>
<li>Structured output generation</li>
</ul>

---

<h2>⚙️ Installation</h2>

Install required dependencies:

```
pip install nltk numpy networkx pdfplumber
```

Download required NLTK resources (automatic on first run).

---

<h2>🚀 Usage</h2>

Basic execution:

```
python summarizer.py document.pdf
```

Custom parameters:

```
python summarizer.py document.pdf --sentences 15 --output summary.txt
```

---

<h2>🏗 System Workflow</h2>

```
PDF Document
     │
     ▼
Text Extraction
     │
Cleaning & Tokenization
     │
Sentence Similarity Matrix
     │
Graph Construction
     │
PageRank Scoring
     │
Top Sentence Selection
     │
Enhanced Summary Output
```

---

<h2>📊 Techniques Applied</h2>

* Natural Language Processing
* TextRank Summarization
* Graph Theory
* Cosine Similarity
* Linguistic Filtering
* Statistical Sentence Ranking

---

<h2>🔮 Future Enhancements</h2>

* Transformer-based summarization models
* Multi-document summarization
* Interactive web dashboard
* Visualization of ranking graphs
* ROUGE-based evaluation metrics

---

<h2>👩‍💻 Author</h2>

**Bhoomika Saxena**
B.Tech — Computer Science (IoT & Intelligent Systems)
AI • Embedded Systems • Applied Research

---


