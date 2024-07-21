import nltk
import numpy as np
from nltk.corpus import stopwords, words
from nltk.cluster.util import cosine_distance
import networkx as nx
import pdfplumber
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')

def pdf_r(file_path):
    with pdfplumber.open(file_path) as pdf_file:
        text = ""
        for page in pdf_file.pages:
            text += page.extract_text()
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9,.\s]', '', text)
    return text

def read_article(file_name):
    article = pdf_r(file_name)
    cleaned_art = clean_text(article)
    sentences = nltk.sent_tokenize(cleaned_art)
    return sentences

def sent_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in nltk.word_tokenize(sent1) if w.lower() not in stopwords]
    sent2 = [w.lower() for w in nltk.word_tokenize(sent2) if w.lower() not in stopwords]
    total_words = list(set(sent1 + sent2))

    vec1 = [0] * len(total_words)
    vec2 = [0] * len(total_words)

    for w in sent1:
        vec1[total_words.index(w)] += 1

    for w in sent2:
        vec2[total_words.index(w)] += 1

    return 1 - cosine_distance(vec1, vec2)

def sim_matrix(sentences, stop_words):
    sim_mat = np.zeros((len(sentences), len(sentences)))
    for i1 in range(len(sentences)):
        for i2 in range(len(sentences)):
            if i1 == i2:
                continue
            sim_mat[i1][i2] = sent_similarity(sentences[i1], sentences[i2], stop_words)
    return sim_mat

def enhance_sentence(sentence):
    enhancements = [
        "This report gives the details about",
        "Let's start with the details about",
        "The well is",
        "We are discussing about",
        "This indicates that",
        "It is noteworthy that",
        "Interestingly,",
        "Furthermore,",
        "In addition to this,",
        "As a result,",
        "Consequently,"
    ]

    if re.search(r'\b(area|wells|structure|production|base)\b', sentence, re.IGNORECASE):
        enhanced_sentence = np.random.choice(enhancements) + " " + sentence
    else:
        enhanced_sentence = sentence

    return enhanced_sentence

def summary_final(file_name, t=5, exclude_words=None, focus_terms=None):
    if exclude_words is None:
        exclude_words = []
    if focus_terms is None:
        focus_terms = []

    stop_words = set(stopwords.words('english'))
    english_words = set(words.words())
    sentences = read_article(file_name)

    filtered_sentences = [
        sentence for sentence in sentences
        if not any(word.lower() in sentence.lower() for word in exclude_words)
    ]

    def is_meaningful(sentence):
        words_in_sentence = nltk.word_tokenize(sentence)
        english_word_count = sum(1 for word in words_in_sentence if word.lower() in english_words)
        return english_word_count / len(words_in_sentence) > 0.5

    meaningful_sentences = [sentence for sentence in filtered_sentences if is_meaningful(sentence)]
    sentence_similarity_matrix = sim_matrix(meaningful_sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(meaningful_sentences)), reverse=True)
    selected_sentences = []
    count = 0
    for score, sentence in ranked_sentences:
        if count >= t:
            break
        if any(term.lower() in sentence.lower() for term in focus_terms):
            selected_sentences.append(sentence)
            count += 1
    if count < t:
        for score, sentence in ranked_sentences:
            if count >= t:
                break
            if sentence not in selected_sentences:
                selected_sentences.append(sentence)
                count += 1
    summary = " ".join(selected_sentences)
    enhanced_summary = " ".join([enhance_sentence(sentence) for sentence in selected_sentences])

    with open("output.txt", "w") as u:
        u.write("Summary:\n")
        u.write(enhanced_summary + "\n")
    print(enhanced_summary)

file_path = "well_student.pdf"
exclude_words = ['Page', 'Figure']
focus_terms = ['area', 'wells', 'structure', 'production', 'base']
summary_final(file_path, t=17, exclude_words=exclude_words, focus_terms=focus_terms)
