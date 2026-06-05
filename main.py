from pathlib import Path

from src.fetcher import PaperFetcher
from src.data_loader import DataLoader
from src.embedder import ThematicEmbedder
from src.analyzer import AlignmentAnalyzer, AIMS_SCOPE
from src.visualizer import AlignmentVisualizer


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "Data" / "papers.csv"
    results_dir = base_dir / "results"

    # 1. Fetch papers from OpenAlex
    # Using PLOS ONE — large journal with thousands of ML papers
    if not data_path.exists():
        fetcher = PaperFetcher(
            search_query=(
                "machine learning artificial intelligence "
                "deep learning neural networks "
                "natural language processing computer vision"
            ),
            start_year=2018,
            end_year=2024,
            target=500,
        )
        fetcher.fetch_and_save(data_path)

    # 2. Load and validate
    loader = DataLoader(data_path, start_year=2018, end_year=2024)
    df = loader.load()

    # 3. Embed with allenai-specter (scientific papers model)
    embedder = ThematicEmbedder(model_name="allenai-specter")
    scope_emb = embedder.encode_single(AIMS_SCOPE)
    paper_embs = embedder.encode(df["abstract"].tolist())

    # 4. Compute alignment scores
    analyzer = AlignmentAnalyzer(aims_scope=AIMS_SCOPE)
    df = analyzer.compute_scores(df, scope_emb, paper_embs)

    # 5. Run BERTopic
    df = analyzer.run_bertopic(df)

    # 6. Temporal drift
    yearly, slope = analyzer.yearly_trend(df)
    analyzer.print_report(df, yearly, slope)

    # 7. Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    df.sort_values("alignment_score", ascending=False).to_csv(
        results_dir / "alignment_scores.csv", index=False
    )
    yearly.to_csv(results_dir / "yearly_alignment.csv", index=False)
    print(f"Results saved to: {results_dir}")

    # 8. Plot everything
    viz = AlignmentVisualizer(results_dir)
    viz.plot_all(df, yearly, slope,
                 topic_model=analyzer.topic_model)

    print("\nDone — check the results/ folder.")


if __name__ == "__main__":
    main()