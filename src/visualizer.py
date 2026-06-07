import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class AlignmentVisualizer:
    """
    Generates all plots for the thematic alignment analysis.
    Covers: score distribution, yearly trend, per-year
    boxplots, category breakdown, and topic distribution.
    """

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", font_scale=1.1)

    def plot_all(
        self,
        df: pd.DataFrame,
        yearly: pd.DataFrame,
        slope: float,
        topic_model=None,
    ) -> None:
        self.plot_histogram(df)
        self.plot_yearly_trend(yearly, slope)
        self.plot_boxplot_by_year(df)
        self.plot_category_by_year(df)
        if topic_model is not None:
            self.plot_topic_distribution(df)
        print("\nAll plots saved to results/")

    def plot_histogram(self, df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(df["alignment_score"], bins=20,
                     kde=True, ax=ax, color="#3B82F6")
        mean = df["alignment_score"].mean()
        q25 = df["alignment_score"].quantile(0.25)
        q75 = df["alignment_score"].quantile(0.75)
        ax.axvline(mean, color="red", linestyle="--",
                   linewidth=1.5, label=f"Mean = {mean:.3f}")
        ax.axvline(q25, color="orange", linestyle=":",
                   linewidth=1.5, label=f"Q25 = {q25:.3f}")
        ax.axvline(q75, color="green", linestyle=":",
                   linewidth=1.5, label=f"Q75 = {q75:.3f}")
        ax.set_title("Alignment Score Distribution",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Cosine Similarity to Aims & Scope")
        ax.set_ylabel("Number of Papers")
        ax.legend()
        plt.tight_layout()
        path = self.results_dir / "alignment_distribution.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path.name}")

    def plot_yearly_trend(
        self, yearly: pd.DataFrame, slope: float
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(yearly["year"], yearly["avg_score"],
                marker="o", linewidth=2,
                color="#3B82F6", label="Mean alignment")
        ax.fill_between(
            yearly["year"],
            yearly["avg_score"] - yearly["std_score"].fillna(0),
            yearly["avg_score"] + yearly["std_score"].fillna(0),
            alpha=0.15, color="#3B82F6", label="±1 std",
        )
        z = np.polyfit(yearly["year"], yearly["avg_score"], 1)
        ax.plot(
            yearly["year"],
            np.poly1d(z)(yearly["year"]),
            linestyle="--", color="red", linewidth=1.5,
            label=f"Trend (slope={slope:.5f})",
        )
        ax.set_title("Yearly Thematic Alignment Trend",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean Alignment Score")
        ax.legend()
        plt.tight_layout()
        path = self.results_dir / "yearly_alignment_trend.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path.name}")

    def plot_boxplot_by_year(self, df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        years = sorted(df["year"].unique())
        data = [
            df[df["year"] == y]["alignment_score"].values
            for y in years
        ]
        ax.boxplot(
            data, labels=years, patch_artist=True,
            boxprops=dict(facecolor="#BFDBFE", color="#1D4ED8"),
            medianprops=dict(color="#1D4ED8", linewidth=2),
        )
        ax.set_title("Alignment Score by Year",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Alignment Score")
        plt.tight_layout()
        path = self.results_dir / "boxplot_by_year.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path.name}")

    def plot_category_by_year(self, df: pd.DataFrame) -> None:
        counts = (
            df.groupby(["year", "alignment_category"])
            .size()
            .unstack(fill_value=0)
        )
        colors = {
            "Low": "#FCA5A5",
            "Medium": "#FCD34D",
            "High": "#6EE7B7",
        }
        fig, ax = plt.subplots(figsize=(10, 5))
        counts.plot(
            kind="bar", stacked=True, ax=ax,
            color=[colors.get(c, "gray") for c in counts.columns],
        )
        ax.set_title("Papers by Alignment Category per Year",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Papers")
        ax.legend(title="Category")
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = self.results_dir / "category_by_year.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path.name}")

    def plot_topic_distribution(self, df: pd.DataFrame) -> None:
        """Plot number of papers per BERTopic topic."""
        if "topic_id" not in df.columns:
            return
        topic_counts = (
            df[df["topic_id"] != -1]
            .groupby(["topic_id", "topic_label"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(15)
        )
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(
            data=topic_counts, y="topic_label", x="count",
            ax=ax, color="#6366F1",
        )
        ax.set_title("Main Topics Discovered by BERTopic",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Number of Papers")
        ax.set_ylabel("Topic")
        plt.tight_layout()
        path = self.results_dir / "topic_distribution.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path.name}")
