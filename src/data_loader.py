import pandas as pd
from pathlib import Path


class DataLoader:
    """
    Loads and validates the papers CSV file.
    Ensures required columns exist and cleans data.
    """

    REQUIRED_COLUMNS = {"title", "abstract", "year"}

    def __init__(self, data_path: Path, start_year: int = 2018,
                 end_year: int = 2024):
        self.data_path = data_path
        self.start_year = start_year
        self.end_year = end_year

    def load(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}"
            )

        df = pd.read_csv(self.data_path)

        missing = self.REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        df = df.dropna(subset=["abstract", "year"]).copy()
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.dropna(subset=["year"])
        df["year"] = df["year"].astype(int)
        df = df[df["year"].between(self.start_year, self.end_year)]
        df = df.reset_index(drop=True)

        print(f"Loaded {len(df)} papers "
              f"({df['year'].min()}–{df['year'].max()})")
        return df