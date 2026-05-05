# Measuring Thematic Alignment Between Journal Scope and Published Articles Using NLP

## 1. Introduction
Scientific journals define an "Aims & Scope" section to guide the type of research they publish. However, it is not always clear whether the actual published articles consistently align with this intended thematic focus.

This project aims to quantitatively evaluate the alignment between a journal's stated scope and the content of its published articles using Natural Language Processing (NLP) techniques. By representing both the journal's scope and article abstracts as vector embeddings, we compute similarity scores to measure thematic consistency.

The goal is to identify:
- how well articles match the intended scope
- whether there are outliers
- whether alignment changes over time

## 2. Methodology

### 2.1 Dataset
A dataset of scientific articles was collected using the arXiv API, focusing on papers related to machine learning. For each paper, the following information was extracted:

- Title
- Abstract
- Year of publication

The dataset contains approximately (replace with your number) articles.

### 2.2 Aims & Scope Definition
A synthetic "Aims & Scope" statement was defined to simulate a journal's thematic focus:

"This journal focuses on machine learning, artificial intelligence, deep learning, neural networks, and data-driven applications."

This text serves as the reference representation of the journal's intended domain.

### 2.3 Text Representation
To convert textual data into numerical form, we used Sentence-BERT (all-MiniLM-L6-v2). This model generates dense vector embeddings that capture semantic meaning.

Each abstract and the Aims & Scope text were encoded into vector representations.

### 2.4 Similarity Measurement
We computed cosine similarity between:

- the embedding of the Aims & Scope
- the embedding of each article abstract

This produced an alignment score between 0 and 1 for each paper.

### 2.5 Analysis
The following analyses were performed:

- Distribution of alignment scores
- Identification of high and low alignment papers
- Detection of outliers
- Temporal analysis of alignment over time

## 3. Experimental Results

### 3.1 Distribution of Alignment
The distribution of alignment scores shows that most papers have moderate to high similarity with the journal scope.

The average alignment score is approximately: **0.385**

### 3.2 Outlier Analysis
We identified papers with very low alignment scores as outliers.

These papers often:

- focus on unrelated domains
- contain broader or interdisciplinary content

This suggests that not all publications strictly follow the journal's thematic focus.

### 3.3 Highly Aligned Papers
Papers with high alignment scores typically:

- directly address machine learning or AI
- use terminology consistent with the scope

This confirms that the method correctly captures thematic similarity.

### 3.4 Temporal Analysis
The analysis of alignment scores over time reveals a clear trend.

👉 Based on your plot:
Alignment scores increase significantly from earlier years to more recent ones.

This suggests that:
- the field has become more specialized
- research is increasingly focused on core topics

However, the low alignment in earlier years may be due to:
- limited data
- greater thematic diversity

## 4. Discussion
The results show that NLP-based similarity measures can effectively capture thematic alignment between journal scope and article content.

However, there are limitations:

- The Aims & Scope was manually defined
- The dataset size may be limited
- Cosine similarity may not capture deeper contextual nuances

Future work could include:

- Topic modeling (e.g., BERTopic)
- Using multiple journals
- Comparing different embedding models

## 5. Conclusion
This project demonstrates that it is possible to quantitatively assess thematic alignment in scientific publishing using NLP techniques.

The approach provides:
- a simple and effective alignment metric
- insights into thematic consistency
- detection of outliers and trends

Such methods can support editorial decisions and improve understanding of research evolution over time.

## 6. Project Structure
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

## 7. Installation
From the project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas matplotlib seaborn sentence-transformers scikit-learn
```

## 8. Run the Pipeline
```bash
python main.py
```

If python points to a different interpreter, use:

```bash
./venv/bin/python main.py
```

## 9. Outputs
The script writes these files to results/:

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

## 10. How to Interpret Results
- High alignment score: paper closely matches journal scope
- Low alignment score: potential thematic outlier
- Negative/declining yearly trend: possible drift away from scope
- Positive/increasing trend: possible consolidation around scope

## 11. Current Scope Definition
The current thematic reference in main.py is:

"This journal publishes research on machine learning, deep learning,
artificial intelligence, neural networks, and data science applications."

## 12. References
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure.
- Reimers, N., and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
- Picascia, S., et al. (2025). The Atlas of Data Science Research.
- Hassan-Montero, Y., et al. (2014). Scimago Journal and Country Rank interface.