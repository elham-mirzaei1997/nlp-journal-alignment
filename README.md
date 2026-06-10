# Measuring Thematic Alignment in Scientific Journals Using NLP

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## Overview

This project investigates whether AI and machine learning
scientific publications align with a predefined journal
Aims & Scope using Natural Language Processing techniques.
The study covers publications from 2018 to 2024, analyzing
thematic coherence, detecting outlier papers, and identifying
long-term conceptual drift.

This work was developed as part of the final project for the
**Natural Language Processing** course, following project
specification P7: *Analyzing Thematic Alignment in Scientific
Journals*.

---

## Research Question

>Do AI and machine learning scientific publications
>consistently align with a predefined journal Aims & Scope
>between 2018 and 2024, and is there measurable thematic
>drift over time?

### Objectives

- Quantify thematic coherence between published articles
  and the journal's Aims & Scope
- Identify outlier papers with low alignment scores
- Detect long-term conceptual evolution through temporal
  drift analysis
- Discover latent research topics using neural topic modeling

---

## Methodology

The pipeline follows four stages as defined in the project
specification:

### Stage 1 — Data Collection
Source: OpenAlex API

The dataset contains scientific papers related to machine learning,
artificial intelligence, and deep learning collected using keyword-based search.

### Stage 2 — Content Modeling

Two complementary NLP methods are applied:

**Method 1 — Sentence-BERT Embeddings**
Scientific Document Embeddings

The project uses allenai-specter, a transformer-based model
trained specifically for scientific paper representation.

> Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence
> Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP.

**Method 2 — BERTopic Topic Modeling**
Latent topics are discovered across all abstracts using
BERTopic, which combines BERT embeddings with a class-based
TF-IDF procedure and HDBSCAN clustering.

> Grootendorst, M. (2022). BERTopic: Neural topic modeling
> with a class-based TF-IDF procedure. arXiv:2203.05794.

### Stage 3 — Alignment Scoring
- **Metric:** Cosine similarity between each paper embedding
  and the Aims & Scope embedding
- **Range:** -1 to +1, where higher values indicate stronger semantic alignment.
- **Thresholds:** Data-driven via Q25/Q75 percentiles
  (no hardcoded values)
- **Outliers:** Z-score threshold of −1.5

### Stage 4 — Analysis
- Score distribution across all papers
- Yearly average alignment (2018–2024)
- Drift detection via linear regression slope
- Category classification: Low / Medium / High alignment
- Qualitative validation of top and bottom ranked papers

---

## Project Structure
thematic_alignment/

│
├── src/  
│   ├── __ init __.py
│   ├── fetcher.py              
│   ├── data_loader.py          
│   ├── embedder.py             
│   ├── analyzer.py             
│   └── visualizer.py   
│
├── notebooks/
│   └── demo.ipynb   
│
├── Data/
│   └── papers.csv  
│
├── results/ 
│   ├── alignment_scores.csv
│   ├── yearly_alignment.csv
│   ├── alignment_distribution.png
│   ├── yearly_alignment_trend.png
│   ├── boxplot_by_year.png
│   ├── category_by_year.png
│   └── topic_distribution.png
│
├── main.py                     
├── requirements.txt
└── README.md

---

## Setup & Reproduction

### Prerequisites
- Python 3.9 or higher
- pip
- Internet connection (for API and model download)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Elham-Mirzaei1997/nlp-journal-alignment.git
cd thematic_alignment

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python3 main.py
```

The script will automatically:
1. Fetch 500 abstracts from OpenAlex API
2. Embed texts using `allenai-specter`
3. Compute cosine similarity alignment scores
4. Run BERTopic topic discovery
5. Generate all results and plots in `results/`

### Run Interactive Demo

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Results


| Metric         | Value     |
| -------------- | --------- |
| Total papers   | 500       |
| Year range     | 2018–2024 |
| Mean alignment | 0.7529    |
| Std deviation  | 0.0675    |
| Maximum score  | 0.9258    |
| Minimum score  | 0.5683    |
| Outliers       | 34        |
| Drift slope    | +0.002820 |



### Output Files

| File | Description |
|------|-------------|
| `alignment_scores.csv` | Per-paper scores, z-scores, categories |
| `yearly_alignment.csv` | Mean alignment per year with std |
| `alignment_distribution.png` | Score histogram with Q25/Q75 |
| `yearly_alignment_trend.png` | Temporal drift with trend line |
| `boxplot_by_year.png` | Score distribution per year |
| `category_by_year.png` | Low/Medium/High breakdown per year |
| `topic_distribution.png` | Top topics from BERTopic |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| OpenAlex over Semantic Scholar | Free, stable, no rate limits |
| `allenai-specter` model | Pre-trained on scientific paper similarity |
| Percentile thresholds (Q25/Q75) | Data-driven, avoids arbitrary cutoffs |
| Z-score outlier detection | Statistically principled approach |
| BERTopic over LDA | Better coherence on short scientific texts |

---

## Code Architecture

The implementation follows an Object-Oriented Programming design.

Responsibilities are separated into independent modules:

- fetcher.py → data collection
- data_loader.py → preprocessing
- embedder.py → embedding generation
- analyzer.py → similarity calculation and evaluation
- visualizer.py → result visualization

This structure improves maintainability and reproducibility.

---

## References

1. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence
   Embeddings using Siamese BERT-Networks. *Proceedings of
   EMNLP-IJCNLP*, 3982–3992.

2. Grootendorst, M. (2022). BERTopic: Neural topic modeling
   with a class-based TF-IDF procedure. *arXiv:2203.05794*.

3. Picascia, S., et al. (2025). The Atlas of Data Science
   Research. *IEEE Access*.

4. Hassan-Montero, Y., et al. (2014). Graphical interface of
   the Scimago Journal and Country Rank. *El profesional de
   la información*, 23(3).

5. Cohan et al. (2020). SPECTER: Document-level Representation Learning 
using Citation-informed Transformers.

---

## AI Usage Disclaimer

AI tools were used to support code organization,
documentation, and debugging.
All results and implementation decisions were reviewed
and validated by the author.

---

## Author

- **Name&Surenamr:** Elham Mirzaei Askarani
- **Course:** Natural Language Processing
- **Project:** P7 — Analyzing Thematic Alignment in Scientific Journals
- **Institution:** *(Università degli Studi di Milano)*