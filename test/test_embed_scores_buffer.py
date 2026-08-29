"""Regression guard for the embedder's eager per-token scores buffer (#6827).

The vendored ``Llama.__init__`` eagerly allocates
``self.scores = np.ndarray((n_batch, n_vocab), dtype=np.single)`` at model load
(the shape is ``(n_batch, n_vocab)`` because ``logits_all`` defaults to False at
construction). The embedding read path never touches ``self.scores`` -- vectors
come from ``llama_get_embeddings_seq`` / ``llama_get_embeddings`` -- so that
array is dead weight whose size is dictated entirely by ``n_batch``. For
Qwen3-Embedding-0.6B (n_vocab 151,936) the old ``n_batch == _N_CTX`` (2048) sized
it at ~1.24 GB; the fix passes a smaller ``n_batch`` (_N_BATCH == 1,536) that
still covers the most tokens a single ``_MAX_EMBED_CHARS``-clipped input can
yield, reclaiming ~311 MB while keeping the produced vectors byte-identical and
``n_ctx`` / ``n_ubatch`` unchanged.

These tests fake the vendored Llama class exactly as ``test/test_embeddings.py``
does (capturing the constructor kwargs); they never load a real model or hit the
network. They FAIL if the construction reverts to ``n_batch == n_ctx``.

In-sandbox this file cannot be collected through the normal ``pytest test/``
path because the heavy root ``test/conftest.py`` imports hypothesis/numpy, which
are not installable under INTEGRATIONS_ONLY. Validate it with the FEAT-001
targeted harness instead: copy this file plus an empty ``conftest.py`` into a
scratch dir and run
``PYTHONPATH=src ~/.pyenv/versions/3.11.15/bin/python -m pytest <scratch>/... \
  -p no:cacheprovider --confcutdir=<scratch> -o addopts='' -q``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import kiro_crew.embeddings as embeddings_mod
from kiro_crew.embeddings import LlamaCppEmbedder


def _make_recording_llama_class(dim: int = embeddings_mod._DEFAULT_DIM):
    """Fake Llama class that records the kwargs its constructor was handed.

    Mirrors ``_make_fake_llama_class`` in test/test_embeddings.py: it never loads
    a real model and returns a fixed-width embedding for ``create_embedding``.
    """

    class _RecordingLlama:
        instances: list = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            type(self).instances.append(self)

        def create_embedding(self, texts):
            return {"data": [{"embedding": [0.1] * dim} for _ in texts]}

    return _RecordingLlama


def _write_model_file(path: Path) -> Path:
    """Write a stand-in GGUF wide enough that model_file_present() accepts it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"g" * (embeddings_mod._GGUF_MIN_BYTES + 100_000))
    return path


def _load_embedder(tmp_path: Path, monkeypatch):
    """Build an embedder wired to a recording fake Llama and load it."""
    fake_cls = _make_recording_llama_class()
    monkeypatch.setattr("kiro_crew.embeddings._load_llama_class", lambda: fake_cls)
    model = _write_model_file(tmp_path / "model.gguf")
    emb = LlamaCppEmbedder(model_path=model)
    assert emb.wait_ready(timeout=5), "fake model failed to load"
    assert fake_cls.instances, "Llama constructor was never called"
    return fake_cls.instances[0].kwargs


def test_embedder_passes_reduced_n_batch(tmp_path: Path, monkeypatch) -> None:
    """The embedder must construct Llama with n_batch == _N_BATCH (< n_ctx)."""
    kwargs = _load_embedder(tmp_path, monkeypatch)
    assert kwargs["n_batch"] == embeddings_mod._N_BATCH
    # The whole point of the fix: n_batch must be strictly below n_ctx, which is
    # what bounds the (n_batch, n_vocab) scores buffer. This assertion FAILS if
    # the construction reverts to n_batch == _N_CTX.
    assert kwargs["n_batch"] < kwargs["n_ctx"]


def test_n_ctx_and_n_ubatch_unchanged(tmp_path: Path, monkeypatch) -> None:
    """Accepted context and physical micro-batch are untouched by the fix."""
    kwargs = _load_embedder(tmp_path, monkeypatch)
    assert kwargs["n_ctx"] == embeddings_mod._N_CTX
    assert kwargs["n_ubatch"] == embeddings_mod._N_UBATCH


def test_n_batch_covers_max_clipped_input(tmp_path: Path, monkeypatch) -> None:
    """n_batch must bound the tokens a single _MAX_EMBED_CHARS input can yield.

    Llama.embed(truncate=True) clips each input to n_batch tokens; if n_batch
    were below the token count of a clipped input the last-token-pooled vector
    would change. The clip is _MAX_EMBED_CHARS chars at the conservative
    ~4 chars/token used to size it, so n_batch must be >= that token count.
    """
    kwargs = _load_embedder(tmp_path, monkeypatch)
    max_tokens_for_clipped_input = embeddings_mod._MAX_EMBED_CHARS // 4
    assert kwargs["n_batch"] >= max_tokens_for_clipped_input
    # llama.cpp requires n_batch >= n_ubatch.
    assert kwargs["n_batch"] >= kwargs["n_ubatch"]


def test_scores_buffer_element_count_is_bounded(tmp_path: Path, monkeypatch) -> None:
    """The reduced n_batch shrinks the (n_batch, n_vocab) scores buffer.

    Emulates the vendored allocation's element count for the shipped vocab and
    asserts the fixed value is well under the old n_ctx-sized footprint, so a
    revert to n_batch == n_ctx would be caught here too.
    """
    kwargs = _load_embedder(tmp_path, monkeypatch)
    n_vocab = 151_936  # Qwen3-Embedding-0.6B
    reduced_elems = kwargs["n_batch"] * n_vocab
    old_elems = kwargs["n_ctx"] * n_vocab
    assert reduced_elems < old_elems
    # float32 => 4 bytes/elem; the reduced buffer must be under 1 GiB (the old
    # one was ~1.16 GiB). Guards against the buffer creeping back toward n_ctx.
    assert reduced_elems * 4 < 1024 ** 3
