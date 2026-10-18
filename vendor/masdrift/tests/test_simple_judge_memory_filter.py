from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "mas_eval"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from mas_eval.simple_judge import retrieve_memory


class _Embedder:
    def embed(self, _texts: list[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


def test_retrieve_memory_filters_by_criterion_before_ranking() -> None:
    bank = {
        "items": [
            {
                "id": "c1-close",
                "metadata": {"criterion": "c1"},
                "embedding": [1.0, 0.0],
            },
            {
                "id": "c2b-only",
                "metadata": {"criterion": "c2b"},
                "embedding": [0.5, 0.0],
            },
        ]
    }

    hits = retrieve_memory(
        query_text="query",
        memory_bank=bank,
        embedder=_Embedder(),
        top_k=3,
        criterion_id="c2b",
    )

    assert [hit["id"] for hit in hits] == ["c2b-only"]


def test_retrieve_memory_can_force_binary_label_coverage() -> None:
    bank = {
        "items": [
            {"id": "positive-a", "label": 1, "embedding": [1.0, 0.0]},
            {"id": "positive-b", "label": 1, "embedding": [0.9, 0.0]},
            {"id": "negative", "label": 0, "embedding": [0.1, 0.0]},
        ]
    }

    hits = retrieve_memory(
        query_text="query",
        memory_bank=bank,
        embedder=_Embedder(),
        top_k=3,
        balance_binary_labels=True,
    )

    assert {hit["label"] for hit in hits} == {0, 1}
    assert len(hits) == 3
