import arxiv
import pandas as pd

def get_papers(query="machine learning", max_results=100):
    search = arxiv.Search(
        query=query,
        max_results=max_results
    )

    papers = []

    for result in search.results():
        papers.append({
            "title": result.title,
            "abstract": result.summary,
            "year": result.published.year
        })

    return pd.DataFrame(papers)


if __name__ == "__main__":
    df = get_papers()
    df.to_csv("data/papers.csv", index=False)
    print("Saved papers to data/papers.csv")