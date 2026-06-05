import time
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path


class PaperFetcher:
    """
    Fetches paper abstracts from the OpenAlex API.
    Free, no authentication required, no rate limits.
    """

    BASE_URL = "https://api.openalex.org/works"

    def __init__(
        self,
        search_query: str = "machine learning deep learning",
        start_year: int = 2018,
        end_year: int = 2024,
        target: int = 500,
    ):
        self.search_query = search_query
        self.start_year = start_year
        self.end_year = end_year
        self.target = target

    @staticmethod
    def _reconstruct_abstract(inv_index: dict) -> str:
        if not inv_index:
            return ""
        words = {}
        for word, positions in inv_index.items():
            for pos in positions:
                words[pos] = word
        return " ".join(words[i] for i in sorted(words))

    def fetch(self) -> pd.DataFrame:
        papers = []
        page = 1
        per_page = 50

        print(f"Fetching from OpenAlex API")
        print(f"Query : {self.search_query}")
        print(f"Years : {self.start_year}-{self.end_year}")
        print(f"Target: {self.target} papers\n")

        with tqdm(total=self.target, desc="Downloading") as pbar:
            while len(papers) < self.target:
                params = {
                    "search": self.search_query,
                    "filter": (
                        f"publication_year:{self.start_year}"
                        f"-{self.end_year},"
                        f"has_abstract:true,"
                        f"type:article"
                    ),
                    "per-page": per_page,
                    "page": page,
                    "select": (
                        "title,abstract_inverted_index,"
                        "publication_year,doi"
                    ),
                }
                try:
                    resp = requests.get(
                        self.BASE_URL,
                        params=params,
                        timeout=30,
                        headers={
                            "User-Agent": "ThematicAlignment/1.0"
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    print(f"\nAPI error on page {page}: {e}")
                    break

                results = data.get("results", [])
                if not results:
                    print("\nNo more results.")
                    break

                for work in results:
                    abstract = self._reconstruct_abstract(
                        work.get("abstract_inverted_index") or {}
                    )
                    if not abstract.strip():
                        continue
                    papers.append({
                        "title": work.get("title", "Unknown"),
                        "abstract": abstract,
                        "year": work.get("publication_year"),
                        "doi": work.get("doi", ""),
                    })
                    pbar.update(1)
                    if len(papers) >= self.target:
                        break

                page += 1
                time.sleep(0.1)

        df = pd.DataFrame(papers[: self.target])
        print(f"\nFetched {len(df)} papers.")
        return df

    def fetch_and_save(self, output_path: Path) -> pd.DataFrame:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.fetch()
        if df.empty:
            raise RuntimeError("No papers fetched.")
        df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
        return df