import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic


AIMS_SCOPE = """
Nature Machine Intelligence publishes high-quality original research
and reviews in machine learning, robotics and AI. Topics include:
deep learning, reinforcement learning, computer vision, natural
language processing, probabilistic methods, AI ethics, fairness,
accountability, transparency, human-computer interaction, and
real-world applications of intelligent systems across science
and society.
"""


class AlignmentAnalyzer:
    """
    Computes thematic alignment between paper abstracts and
    a journal's Aims & Scope using two complementary methods:

    1. Sentence-BERT cosine similarity (Reimers & Gurevych, 2019)
    2. BERTopic topic modeling (Grootendorst, 2022)

    Research Question:
        Do published papers consistently align with the journal's
        stated Aims & Scope, and does alignment drift over time?
    """

    def __init__(self, aims_scope: str = AIMS_SCOPE):
        self.aims_scope = aims_scope
        self.low_thresh = None
        self.high_thresh = None
        self.topic_model = None

    # ── Embedding-based alignment ─────────────────────────────

    def compute_scores(
        self,
        df: pd.DataFrame,
        scope_embedding: np.ndarray,
        paper_embeddings: np.ndarray,
    ) -> pd.DataFrame:
        """Compute cosine similarity scores for each paper."""
        df = df.copy()

        df["alignment_score"] = cosine_similarity(
            scope_embedding, paper_embeddings
        )[0]

        # Z-score for outlier detection
        mean = df["alignment_score"].mean()
        std = df["alignment_score"].std(ddof=1)
        df["alignment_zscore"] = (
            (df["alignment_score"] - mean) / std
            if std > 0 else 0.0
        )

        # Data-driven thresholds — no hardcoding
        self.low_thresh = df["alignment_score"].quantile(0.25)
        self.high_thresh = df["alignment_score"].quantile(0.75)

        df["alignment_category"] = pd.cut(
            df["alignment_score"],
            bins=[-np.inf, self.low_thresh, self.high_thresh, np.inf],
            labels=["Low", "Medium", "High"],
        )
        df["is_outlier"] = df["alignment_zscore"] <= -1.5

        return df

    # ── BERTopic topic modeling ───────────────────────────────

    def run_bertopic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit BERTopic on abstracts to discover latent topics.

        Reference: Grootendorst, M. (2022). BERTopic: Neural topic
        modeling with a class-based TF-IDF procedure.
        arXiv:2203.05794.
        """
        print("\nRunning BERTopic on abstracts...")
        abstracts = df["abstract"].tolist()

        self.topic_model = BERTopic(
            language="english",
            calculate_probabilities=True,
            verbose=False,
            nr_topics="auto",
            min_topic_size=5,
        )

        topics, probs = self.topic_model.fit_transform(abstracts)
        df = df.copy()
        df["topic_id"] = topics

        # Get topic labels
        topic_info = self.topic_model.get_topic_info()
        topic_labels = dict(
            zip(topic_info["Topic"], topic_info["Name"])
        )
        df["topic_label"] = df["topic_id"].map(topic_labels)

        print(f"Discovered {len(topic_info) - 1} topics "
              f"(excluding outlier topic -1)")
        return df

    def get_topic_summary(self) -> pd.DataFrame:
        """Return top topics discovered by BERTopic."""
        if self.topic_model is None:
            return pd.DataFrame()
        return self.topic_model.get_topic_info().head(15)

    # ── Temporal drift analysis ───────────────────────────────

    def yearly_trend(self, df: pd.DataFrame) -> tuple:
        """Compute mean alignment per year and drift slope."""
        yearly = (
            df.groupby("year")["alignment_score"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={
                "mean": "avg_score",
                "std": "std_score",
                "count": "n_papers",
            })
        )
        slope = (
            float(np.polyfit(
                yearly["year"], yearly["avg_score"], 1
            )[0])
            if len(yearly) >= 3
            else 0.0
        )
        return yearly, slope

    # ── Summary & reporting ───────────────────────────────────

    def summary_stats(self, df: pd.DataFrame) -> dict:
        return {
            "n_papers": len(df),
            "year_min": int(df["year"].min()),
            "year_max": int(df["year"].max()),
            "mean_score": round(float(
                df["alignment_score"].mean()), 4),
            "std_score": round(float(
                df["alignment_score"].std()), 4),
            "max_score": round(float(
                df["alignment_score"].max()), 4),
            "min_score": round(float(
                df["alignment_score"].min()), 4),
            "low_threshold_q25": round(float(self.low_thresh), 4),
            "high_threshold_q75": round(float(self.high_thresh), 4),
            "n_low": int(
                (df["alignment_category"] == "Low").sum()),
            "n_medium": int(
                (df["alignment_category"] == "Medium").sum()),
            "n_high": int(
                (df["alignment_category"] == "High").sum()),
            "n_outliers": int(df["is_outlier"].sum()),
        }

    def print_report(
        self,
        df: pd.DataFrame,
        yearly: pd.DataFrame,
        slope: float,
    ) -> None:
        stats = self.summary_stats(df)

        print("\n" + "=" * 60)
        print("     THEMATIC ALIGNMENT ANALYSIS REPORT")
        print("=" * 60)
        for k, v in stats.items():
            print(f"  {k:<30}: {v}")

        trend = "improving" if slope > 0 else "declining"
        print(f"  {'drift_slope':<30}: {slope:.6f} ({trend})")

        print("\n  Yearly alignment:")
        print(yearly.to_string(index=False))

        print("\n  Top 5 most aligned papers:")
        for _, r in df.nlargest(5, "alignment_score").iterrows():
            print(f"    [{r['year']}] {r['alignment_score']:.4f}"
                  f" — {str(r['title'])[:65]}")

        print("\n  Top 5 least aligned papers:")
        for _, r in df.nsmallest(5, "alignment_score").iterrows():
            print(f"    [{r['year']}] {r['alignment_score']:.4f}"
                  f" — {str(r['title'])[:65]}")

        outliers = df[df["is_outlier"]]
        print(f"\n  Outliers (z <= -1.5): {len(outliers)}")
        for _, r in outliers.iterrows():
            print(f"    z={r['alignment_zscore']:.2f}"
                  f" — {str(r['title'])[:65]}")

        # BERTopic summary if available
        if self.topic_model is not None:
            print("\n  Top discovered topics (BERTopic):")
            topic_info = self.get_topic_summary()
            for _, row in topic_info.iterrows():
                if row["Topic"] == -1:
                    continue
                print(f"    Topic {row['Topic']:>2}: "
                      f"{row['Name'][:60]}")

        print("=" * 60)
