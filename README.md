<h1 align="center">📄 Intelligent PDF Text Summarizer</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-%233776AB.svg?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/NLP-NLTK-%237B68EE.svg?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/GraphRank-TextRank-%23FF9800.svg?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PDF-Processing-%23E53935.svg?style=for-the-badge"/>
</p>

<hr/>

<h2>📌 Overview</h2>
<p>
An intelligent NLP-based document summarization system that extracts text from PDF files,
ranks sentences using a TextRank graph algorithm, and generates concise summaries.
The system supports domain-focused summarization, sentence enhancement,
and robust filtering for meaningful content extraction.
</p>

---

<h2>🧠 Core Features</h2>

<ul>
<li>PDF text extraction</li>
<li>Sentence ranking using PageRank graph model</li>
<li>Focus-term prioritization</li>
<li>Noise filtering & language validation</li>
<li>Context-aware sentence enhancement</li>
<li>CLI execution support</li>
<li>Professional logging</li>
</ul>

---

<h2>⚙️ Installation</h2>

```
pip install nltk numpy networkx pdfplumber
```

Run once:

```
python summarizer.py sample.pdf
```

---

<h2>🚀 Usage</h2>

```
python summarizer.py document.pdf --sentences 15 --output summary.txt
```

---

<h2>🏗 Architecture</h2>

```
PDF Input
   │
   ▼
Text Extraction
   │
Cleaning & Tokenization
   │
Similarity Matrix
   │
Graph Construction
   │
PageRank Scoring
   │
Sentence Selection
   │
Enhanced Summary Output
```

---

<h2>📊 Techniques Used</h2>

* Natural Language Processing
* Cosine Similarity
* Graph-Based Ranking
* TextRank Summarization
* Linguistic Filtering

---

<h2>🔮 Future Scope</h2>

* Transformer-based summarization
* Multi-document aggregation
* Web UI interface
* Domain-adaptive fine tuning
* GPU acceleration

---

<h2>👩‍💻 Author</h2>

**Bhoomika Saxena**
AI • IoT • Embedded Systems • Applied Research

---

<h2>📜 License</h2>

Academic and research use.

<hr/>
<p align="center">
⭐ Star the repository if this helps your research
</p>
