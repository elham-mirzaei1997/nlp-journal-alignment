# Measuring Thematic Alignment Between Journal Scope and Published Articles Using NLP

## 1. Introduction

Scientific journals define an “Aims & Scope” section to describe the type of research they intend to publish. However, it is not always clear whether the articles published in a journal consistently follow this thematic focus.

This project aims to evaluate the alignment between a journal’s stated scope and the content of its published articles using Natural Language Processing (NLP) techniques.

By representing both the journal’s scope and article abstracts as vector embeddings, we compute similarity scores to measure thematic consistency.

The main objectives of this study are:
- to quantify how well articles match the intended scope
- to identify outlier papers that deviate from the theme
- to analyze how alignment evolves over time

## 2. Methodology

### 2.1 Dataset

A dataset of scientific articles was collected using the arXiv API. For each paper, the following information was extracted: title, abstract, and year of publication.

The dataset used in this project contains X articles (replace with your number).

---

### 2.2 Aims & Scope Definition

A synthetic “Aims & Scope” statement was defined to represent the thematic focus of a journal:

"This journal focuses on machine learning, artificial intelligence, deep learning, neural networks, and data-driven applications."

---

### 2.3 Text Representation

To convert textual data into numerical form, we used Sentence-BERT (all-MiniLM-L6-v2). This model generates dense vector embeddings that capture semantic meaning.

Each abstract and the Aims & Scope text were encoded into vector representations.

---

### 2.4 Similarity Measurement

We computed cosine similarity between the embedding of the Aims & Scope and the embedding of each article abstract.

This resulted in an alignment score between 0 and 1 for each paper.

---

### 2.5 Analysis

We performed the following analyses:
- distribution of alignment scores
- identification of high and low alignment papers
- detection of outliers
- temporal analysis of alignment over time

## 3. Results

### 3.1 Overall Alignment

The average alignment score is 0.385, with values ranging from 0.07 to 0.62. This indicates a moderate level of thematic consistency between the articles and the journal’s scope.

---

### 3.2 Outlier Analysis

We identified low-alignment papers as potential outliers. One paper (Paper4, 2020) shows a very low alignment score (0.07), indicating a strong thematic mismatch.

This suggests that not all articles strictly follow the journal’s intended focus.

---

### 3.3 Distribution of Scores

The distribution of alignment scores shows that most papers fall within a moderate range. However, there is significant variability, indicating differences in thematic relevance.

(👉 Here you can mention your histogram or boxplot)

---

### 3.4 Temporal Analysis

The analysis of alignment scores over time shows a clear increasing trend.

Alignment scores increased from 0.07 in 2020 to 0.62 in 2023, with an estimated slope of 0.18 per year.

This suggests that more recent publications are increasingly aligned with the journal’s thematic scope.

## 4. Discussion

The results demonstrate that NLP-based similarity measures can effectively capture thematic alignment between journal scope and article content.

However, several limitations should be considered. First, the dataset is relatively small, which may affect the reliability of the results. In particular, some years contain only a single paper, which may distort the trend analysis.

Second, the Aims & Scope was manually defined, which may introduce bias.

Finally, cosine similarity captures general semantic similarity but may not fully reflect deeper contextual relationships.

Future work could include:
- using larger datasets
- applying topic modeling methods such as BERTopic
- comparing multiple journals

## 5. Conclusion

This project demonstrates that it is possible to quantitatively assess thematic alignment in scientific publications using NLP techniques.

The approach provides a simple and effective way to measure alignment, identify outliers, and analyze trends over time.

Despite its limitations, the method offers useful insights into how closely published articles follow a journal’s intended scope.

## 6. AI Assistance

Parts of this project were developed with the assistance of OpenAI’s ChatGPT. All outputs were reviewed and adapted by the author.
