# Measuring Thematic Alignment Between Journal Scope and Published Articles Using NLP

## Overview
This project measures how closely published article abstracts align with a journal's stated scope using NLP. It compares a journal description with article text and produces alignment scores for each paper. The results help identify thematic fit, outliers, and changes over time.

## Objective
The goal is to quantify journal-article alignment in a simple and reproducible way. This helps show whether the published papers match the intended scope and where the strongest mismatches appear.

## Method
- Sentence-BERT converts the journal scope and each article abstract into embeddings.
- Cosine similarity compares those embeddings and produces an alignment score for each article.
- Higher scores mean the article is closer to the journal scope.
- Cosine similarity is used as the evaluation metric to measure alignment between texts.

## Dataset
The dataset contains 5 articles with the following fields:
- Title
- Abstract
- Year of publication

## Project Structure
```
NLP project/
├── Data/
│   └── papers.csv
├── results/
├── src/
├── notebooks/
├── main.py
└── README.md
```

## Installation
From the project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas matplotlib seaborn sentence-transformers scikit-learn
```

## Run Instructions
```bash
python main.py
```

If python points to a different interpreter, use:

```bash
./venv/bin/python main.py
```

## Outputs
The script writes these files to `results/`:

CSV files:
- alignment_scores.csv
- yearly_alignment.csv
- output.csv
- final_results.csv

Plots:
- alignment_distribution.png
- yearly_alignment_trend.png
- alignment_over_time.png
- alignment_boxplot.png

Console output includes:
- Top and least aligned papers
- Outlier table
- Yearly averages
- Drift slope
- Summary statistics (average, min, max)

## Interpretation of Results
- High alignment score: paper closely matches journal scope
- Low alignment score: potential thematic outlier
- Negative/declining yearly trend: possible drift away from scope
- Positive/increasing trend: possible consolidation around scope

## References
- Reimers, N., and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure.