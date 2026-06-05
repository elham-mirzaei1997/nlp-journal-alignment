import numpy as np
from sentence_transformers import SentenceTransformer


class ThematicEmbedder:
    """
    Computes sentence embeddings using Sentence-BERT
    (all-mpnet-base-v2), following:

    Reimers, N., & Gurevych, I. (2019). Sentence-BERT:
    Sentence Embeddings using Siamese BERT-Networks.
    EMNLP-IJCNLP, 3982-3992.
    """

    # Sentence-BERT model as cited in project references
    DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"

    def __init__(self, model_name: str = None):
        model_name = model_name or self.DEFAULT_MODEL
        print(f"Loading Sentence-BERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts into embedding vectors."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text (e.g. Aims & Scope)."""
        return self.model.encode([text], convert_to_numpy=True)
