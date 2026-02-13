"""
PDF Text Summarizer

Extracts text from a PDF, ranks sentences using TextRank,
and produces a focus-aware summary.
"""

import argparse
import logging
import re
from pathlib import Path
from typing import List

import nltk
import numpy as np
import networkx as nx
import pdfplumber
from nltk.corpus import stopwords, words
from nltk.cluster.util import cosine_distance


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("words", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
ENGLISH_WORDS = set(words.words())


# PDF text extraction
def extract_pdf_text(file_path: Path) -> str:
    logging.info(f"Reading PDF: {file_path}")
    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "

    return text


# Text normalization
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9,. ]", "", text)
    return text.strip()


def tokenize_sentences(text: str) -> List[str]:
    return nltk.sent_tokenize(text)


# Sentence similarity via cosine distance
def sentence_similarity(sent1: str, sent2: str) -> float:
    tokens1 = [
        w.lower()
        for w in nltk.word_tokenize(sent1)
        if w.lower() not in STOP_WORDS
    ]
    tokens2 = [
        w.lower()
        for w in nltk.word_tokenize(sent2)
        if w.lower() not in STOP_WORDS
    ]

    vocab = list(set(tokens1 + tokens2))
    vec1 = np.zeros(len(vocab))
    vec2 = np.zeros(len(vocab))

    for w in tokens1:
        vec1[vocab.index(w)] += 1
    for w in tokens2:
        vec2[vocab.index(w)] += 1

    return 1 - cosine_distance(vec1, vec2)


# Similarity matrix construction
def build_similarity_matrix(sentences: List[str]) -> np.ndarray:
    size = len(sentences)
    matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if i != j:
                matrix[i][j] = sentence_similarity(sentences[i], sentences[j])

    return matrix


# Filter low-quality or unwanted sentences
def meaningful(sentence: str) -> bool:
    tokens = nltk.word_tokenize(sentence)
    if not tokens:
        return False

    english_count = sum(1 for w in tokens if w.lower() in ENGLISH_WORDS)
    return english_count / len(tokens) > 0.5


def filter_sentences(sentences: List[str], exclude_words: List[str]) -> List[str]:
    return [
        s for s in sentences
        if meaningful(s)
        and not any(word.lower() in s.lower() for word in exclude_words)
    ]


ENHANCEMENTS = [
    "This report highlights that",
    "It is observed that",
    "Importantly,",
    "Furthermore,",
    "In addition,",
    "Consequently,",
    "This indicates that"
]


# Optional summary phrasing enhancement
def enhance(sentence: str) -> str:
    if re.search(r"\b(area|wells|structure|production|base)\b", sentence, re.I):
        return f"{np.random.choice(ENHANCEMENTS)} {sentence}"
    return sentence


# TextRank-based summarization
def summarize(
    file_path: Path,
    top_n: int,
    exclude_words: List[str],
    focus_terms: List[str],
    output_file: Path
):

    text = extract_pdf_text(file_path)
    text = clean_text(text)
    sentences = tokenize_sentences(text)

    sentences = filter_sentences(sentences, exclude_words)

    matrix = build_similarity_matrix(sentences)
    graph = nx.from_numpy_array(matrix)
    scores = nx.pagerank(graph)

    ranked = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)),
        reverse=True
    )

    selected = []

    for _, s in ranked:
        if len(selected) >= top_n:
            break
        if any(term.lower() in s.lower() for term in focus_terms):
            selected.append(s)

    for _, s in ranked:
        if len(selected) >= top_n:
            break
        if s not in selected:
            selected.append(s)

    enhanced = " ".join(enhance(s) for s in selected)

    output_file.write_text(enhanced)
    logging.info(f"Summary saved to {output_file}")

    print("\nSummary:\n")
    print(enhanced)


# Entry point
if __name__ == "__main__":

    pdf_file_path = "/Project Report b1.pdf"
    num_sentences = 10
    output_filename = "summary.txt"

    class Args:
        def __init__(self, pdf, sentences, output):
            self.pdf = pdf
            self.sentences = sentences
            self.output = output

    args = Args(pdf_file_path, num_sentences, output_filename)

    summarize(
        Path(args.pdf),
        args.sentences,
        exclude_words=["Page", "Figure"],
        focus_terms=["area", "wells", "structure", "production", "base"],
        output_file=Path(args.output)
    )
