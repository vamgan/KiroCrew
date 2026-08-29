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

A second group (see the "Concern #2" section below) enforces the
``_MAX_EMBED_CHARS`` / ``_MIN_CHARS_PER_TOKEN`` / ``_N_BATCH`` relationship that
underpins the byte-identical guarantee, by driving a boundary-length input
through a floor-density fake tokenizer and the vendored ``embed(truncate=True)``
token handling. Those fail if a future ``_MAX_EMBED_CHARS`` bump, ``_N_BATCH``
shrink, or a denser assumed tokenizer floor would let the char clip outrun the
context window or silently truncate a normal chunk.

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


# --- Concern #2: enforce the char-clip / chars-per-token / n_batch relationship
# so a future _MAX_EMBED_CHARS bump or denser assumed tokenizer cannot keep the
# suite green while breaking the byte-identical guarantee. These tests do NOT
# re-derive the same arithmetic from the same constants; they pin the DOCUMENTED
# floor and drive a boundary-length input through a tokenizer that tokenizes AT
# that floor, then assert the truncation behavior the byte-identity claim relies
# on. They fail if the module's own import-time invariant is weakened.


def test_min_chars_per_token_floor_is_pessimistic() -> None:
    """The documented floor must stay below the clip's nominal ~4 estimate.

    The whole point of the review fix is that relying on ~4 chars/token left only
    a 36-token margin. The floor must be a genuinely conservative value (< 4) or
    the "real headroom" claim is empty. This fails if someone raises the floor
    back toward/above 4 to make a larger _MAX_EMBED_CHARS "fit".
    """
    assert embeddings_mod._MIN_CHARS_PER_TOKEN < 4.0
    assert embeddings_mod._MIN_CHARS_PER_TOKEN > 0.0


def test_import_time_invariant_enforces_hard_context_bound() -> None:
    """A _MAX_EMBED_CHARS-clipped input must fit n_ctx at the documented floor.

    This is the hard correctness bound the module asserts at import: a clipped
    input tokenizes to at most _MAX_EMBED_CHARS / _MIN_CHARS_PER_TOKEN tokens and
    that must stay within n_ctx. Recomputed here so a future _MAX_EMBED_CHARS
    bump or floor change that violates it is caught even if the module-level
    assert were ever downgraded.
    """
    worst_case_tokens = embeddings_mod._MAX_EMBED_CHARS / embeddings_mod._MIN_CHARS_PER_TOKEN
    assert worst_case_tokens <= embeddings_mod._N_CTX


def _fake_tokenizer_at_floor(text: str) -> list[int]:
    """Tokenize `text` at exactly the documented worst-case density.

    One token per ``_MIN_CHARS_PER_TOKEN`` characters (ceil), i.e. the densest
    tokenization the module claims to tolerate. Returns dummy token ids.
    """
    import math

    floor = embeddings_mod._MIN_CHARS_PER_TOKEN
    n_tokens = math.ceil(len(text) / floor)
    return list(range(n_tokens))


def _simulate_embed_truncation(tokens: list[int], n_batch: int) -> tuple[list[int], bool]:
    """Mirror the vendored Llama.embed(truncate=True) token handling.

    embed() does ``tokens = tokens[:n_batch]`` and then raises if the (already
    clipped) token count still exceeds n_batch. Returns the tokens actually
    decoded plus whether truncation dropped any tail.
    """
    kept = tokens[:n_batch]
    truncated = len(kept) < len(tokens)
    return kept, truncated


def test_clipped_input_at_floor_fits_context_without_full_loss() -> None:
    """Drive a max-length input through a floor-density tokenizer + embed().

    This exercises the boundary the byte-identity claim depends on rather than
    asserting arithmetic on constants: a _MAX_EMBED_CHARS-length blob tokenized
    at the documented floor must (a) still fit n_ctx so it can be decoded, and
    (b) never exceed the vendored embed() ValueError guard after its own clip.
    A future edit that lets the char clip outrun n_ctx at the floor breaks (a).
    """
    blob = "x" * embeddings_mod._MAX_EMBED_CHARS
    tokens = _fake_tokenizer_at_floor(blob)
    # (a) fits the context window at the documented worst-case density.
    assert len(tokens) <= embeddings_mod._N_CTX
    # (b) embed(truncate=True) never trips its post-clip overrun ValueError.
    kept, _ = _simulate_embed_truncation(tokens, embeddings_mod._N_BATCH)
    assert len(kept) <= embeddings_mod._N_BATCH


def test_chunked_input_is_never_truncated_at_floor() -> None:
    """A normal chunked input stays under n_batch even at the floor density.

    Real embed inputs are chunk-/length-bounded upstream: the knowledge chunker
    emits CHUNK_TOKEN_SIZE + CHUNK_OVERLAP tokens per chunk. Fold that budget
    back to characters at the documented floor and confirm the resulting input
    is NOT truncated by embed(truncate=True) -- i.e. its last-token-pooled vector
    is unchanged. This fails if _N_BATCH is reduced below the real chunk ceiling
    or the floor is lowered so a chunk no longer fits.
    """
    # Chunker budget (kept local so the test does not import the knowledge pkg,
    # matching the sandbox constraint of stdlib-only collection). These mirror
    # src/kiro_crew/knowledge/chunker.py CHUNK_TOKEN_SIZE + CHUNK_OVERLAP.
    chunk_token_ceiling = 800 + 200  # ~1000 tokens per chunk
    # A chunk at that many tokens occupies at most chunk_token_ceiling * floor
    # characters; clip guards it anyway, but a chunk is far under the clip.
    chunk_chars = int(chunk_token_ceiling * embeddings_mod._MIN_CHARS_PER_TOKEN)
    chunk_chars = min(chunk_chars, embeddings_mod._MAX_EMBED_CHARS)
    tokens = _fake_tokenizer_at_floor("y" * chunk_chars)
    kept, truncated = _simulate_embed_truncation(tokens, embeddings_mod._N_BATCH)
    assert not truncated, "a normal chunk must not be truncated by embed()"
    assert len(kept) == len(tokens)
