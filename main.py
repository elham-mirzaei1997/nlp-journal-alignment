from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	data_path = base_dir / "Data" / "papers.csv"
	results_dir = base_dir / "results"
	results_dir.mkdir(parents=True, exist_ok=True)

	# Step 1: Load data and validate expected structure.
	df = pd.read_csv(data_path)
	required_columns = {"title", "abstract", "year"}
	missing = required_columns.difference(df.columns)
	if missing:
		raise ValueError(f"Missing required columns in dataset: {sorted(missing)}")

	df = df.dropna(subset=["abstract", "year"]).copy()
	df["year"] = pd.to_numeric(df["year"], errors="coerce")
	df = df.dropna(subset=["year"]).copy()
	df["year"] = df["year"].astype(int)

	# Step 2: Define the Aims & Scope text as thematic reference.
	aims_scope = """
	This journal publishes research on machine learning, deep learning,
	artificial intelligence, neural networks, and data science applications.
	"""

	# Step 3: Get embeddings and compute alignment scores.
	model = SentenceTransformer("all-MiniLM-L6-v2")
	aim_embedding = model.encode([aims_scope])
	paper_embeddings = model.encode(df["abstract"].tolist())

	# Step 4: Compute article-level alignment scores.
	df["alignment_score"] = cosine_similarity(aim_embedding, paper_embeddings)[0]

	# Step 5: Detect low-alignment outliers using a z-score threshold.
	score_mean = df["alignment_score"].mean()
	score_std = df["alignment_score"].std(ddof=0)
	if score_std == 0:
		df["alignment_zscore"] = 0.0
	else:
		df["alignment_zscore"] = (df["alignment_score"] - score_mean) / score_std
	df["is_outlier_low_alignment"] = df["alignment_zscore"] <= -1.0

	# Step 6: Analyze thematic drift by year via average alignment trend.
	yearly = (
		df.groupby("year", as_index=False)["alignment_score"]
		.mean()
		.sort_values("year")
	)
	if len(yearly) >= 2:
		drift_slope = (yearly["alignment_score"].iloc[-1] - yearly["alignment_score"].iloc[0]) / (
			yearly["year"].iloc[-1] - yearly["year"].iloc[0]
		)
	else:
		drift_slope = 0.0

	# Step 7: Save detailed and aggregate outputs.
	article_out_path = results_dir / "alignment_scores.csv"
	yearly_out_path = results_dir / "yearly_alignment.csv"
	df.sort_values("alignment_score", ascending=False).to_csv(article_out_path, index=False)
	yearly.to_csv(yearly_out_path, index=False)
	df.to_csv(results_dir / "output.csv", index=False)
	df.to_csv(results_dir / "final_results.csv", index=False)

	# Step 8: Plot distribution and long-term trend.
	plt.figure(figsize=(8, 4))
	plt.hist(df["alignment_score"], bins=8)
	plt.title("Alignment Score Distribution")
	plt.xlabel("Cosine Similarity to Aims & Scope")
	plt.ylabel("Count")
	plt.tight_layout()
	plt.savefig(results_dir / "alignment_distribution.png", dpi=150)
	plt.close()

	if not yearly.empty:
		plt.figure(figsize=(8, 4))
		plt.plot(yearly["year"], yearly["alignment_score"], marker="o")
		plt.title("Yearly Thematic Alignment Trend")
		plt.xlabel("Year")
		plt.ylabel("Mean Alignment Score")
		plt.tight_layout()
		plt.savefig(results_dir / "yearly_alignment_trend.png", dpi=150)
		plt.close()

	plt.figure(figsize=(8, 4))
	df.groupby("year")["alignment_score"].mean().plot(marker="o")
	plt.title("Alignment Over Time")
	plt.xlabel("Year")
	plt.ylabel("Mean Alignment Score")
	plt.tight_layout()
	plt.savefig(results_dir / "alignment_over_time.png", dpi=150)
	plt.close()

	plt.figure(figsize=(8, 4))
	sns.boxplot(x=df["alignment_score"])
	plt.title("Alignment Score Boxplot")
	plt.tight_layout()
	plt.savefig(results_dir / "alignment_boxplot.png", dpi=150)
	plt.close()

	# Step 9: Print comprehensive analysis.
	print("\n" + "="*60)
	print("THEMATIC ALIGNMENT ANALYSIS")
	print("="*60)

	print("\nTop aligned papers:")
	print(df.sort_values("alignment_score", ascending=False)[["title", "year", "alignment_score"]])

	print("\nLeast aligned papers:")
	print(df.sort_values("alignment_score")[["title", "year", "alignment_score"]])

	print("\nTop 5 MOST aligned papers:")
	print(df.sort_values("alignment_score", ascending=False).head(5)[["title", "alignment_score"]])

	print("\nTop 5 LEAST aligned papers (outliers):")
	print(df.sort_values("alignment_score").head(5)[["title", "alignment_score"]])

	outliers = df[df["is_outlier_low_alignment"]]
	print("\nLow-alignment outliers (z <= -1.0):")
	if outliers.empty:
		print("No outliers detected with current threshold.")
	else:
		print(outliers[["title", "year", "alignment_score", "alignment_zscore"]])

	# Threshold-based classification
	low_threshold = 0.4
	high_threshold = 0.7
	low_aligned = df[df["alignment_score"] < low_threshold]
	high_aligned = df[df["alignment_score"] > high_threshold]

	print("\nNumber of LOW aligned papers (<0.4):", len(low_aligned))
	print("Number of HIGH aligned papers (>0.7):", len(high_aligned))

	print("\nYearly average alignment:")
	print(yearly)
	print(f"\nEstimated drift slope (score/year): {drift_slope:.4f}")

	print("\nAlignment statistics:")
	print("Average alignment:", df["alignment_score"].mean())
	print("Max alignment:", df["alignment_score"].max())
	print("Min alignment:", df["alignment_score"].min())

	print("\nPapers per year:")
	print(df.groupby("year")["alignment_score"].count())

	print(f"\nSaved article-level results: {article_out_path}")
	print(f"Saved yearly trend results: {yearly_out_path}")
	print("="*60)


if __name__ == "__main__":
	main()