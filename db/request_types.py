from pathlib import Path

import lancedb
import pandas as pd
from fastembed import TextEmbedding
from lancedb.embeddings import TextEmbeddingFunction, get_registry, register

DATA = Path(__file__).resolve().parent.parent / "data"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@register("fastembed")
class FastEmbed(TextEmbeddingFunction):
    name: str = EMBED_MODEL

    def ndims(self) -> int:
        return 384

    def _model(self) -> TextEmbedding:
        if not hasattr(self, "_cached"):
            self._cached = TextEmbedding(self.name, cache_dir=str(DATA / "models"))
        return self._cached

    def generate_embeddings(self, texts) -> list:
        return [v for v in self._model().embed(list(texts), batch_size=64)]

    def compute_query_embeddings(self, query: str, *args, **kwargs) -> list:
        return self.generate_embeddings([QUERY_PREFIX + query])


embedder = get_registry().get("fastembed").create()


def search(text: str, k: int = 10) -> pd.DataFrame:
    tbl = lancedb.connect(DATA / "lancedb").open_table("request_types")
    return tbl.search(text, query_type="hybrid").limit(k).to_pandas()
