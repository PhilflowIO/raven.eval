"""Every self-hosted result names the serving setup that answered it — and only one.

Pins the harness side of ``raven_asr.serving``: the adapter pins the first
response's engine / recipe / batching / decoding / precision, a later different block aborts the run
like a served-model mismatch (exit 3, nothing recorded), a resumed subset whose
earlier passes recorded another block is refused before its first request, a
server that reports none records ``serving: null``, and ``promote`` / ``verify``
carry and check the block without failing artifacts that predate it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("httpx")
pytest.importorskip("tqdm")

import httpx
from test_resume import MODEL, SUBSET, _LazyLoader, _samples, _wire

from raven_asr import promote as promote_mod
from raven_asr import reproduce, runner, serving
from raven_asr.adapters.vllm_openai import VllmOpenAIAdapter

BLOCK_A = {
    "engine": "vllm 0.28.0",
    "recipe": {"id": "primeline-whisper-large-v3-german", "sha256": "a" * 64},
    "batching": {"max_num_seqs": 16},
    "decoding": {"mode": "greedy", "max_tokens": 1024},
    "precision": {
        "weights": "fp16", "activations": "fp16", "scheme": "none", "tensor_parallel": 1,
        "checkpoint": {"source": "published", "dtype": "bf16"},
    },
}
BLOCK_B = {**BLOCK_A, "engine": "vllm 0.29.0"}


class _Server:
    """A fake endpoint: transcribes every clip, reports ``blocks[i]`` on request i."""

    def __init__(self, blocks: list[dict[str, Any] | None]) -> None:
        self.blocks = blocks
        self.n = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        block = self.blocks[min(self.n, len(self.blocks) - 1)]
        self.n += 1
        x_raven: dict[str, Any] = {"model": MODEL}
        if block is not None:
            x_raven.update(block)
        return httpx.Response(200, json={"text": "hallo welt", "x_raven": x_raven})


def _adapter(monkeypatch: pytest.MonkeyPatch, server: _Server) -> VllmOpenAIAdapter:
    monkeypatch.setenv("VLLM_BENCH_URL", "http://bench.invalid/v1")
    return VllmOpenAIAdapter(
        provider_id="t", model_id=MODEL, base_url_env="VLLM_BENCH_URL",
        transport=httpx.MockTransport(server),
    )


def _run(out: Path, concurrency: int = 1) -> None:
    runner.run(model_key=MODEL, subsets=[SUBSET], limit=None, out_dir=out,
               concurrency=concurrency)


def _row(out: Path) -> dict[str, Any]:
    (row,) = json.loads((out / "summary.json").read_text())["results"]
    return row


def _passes(out: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (out / f"throughput_{SUBSET}.jsonl").read_text().splitlines()]


# ── Recording ────────────────────────────────────────────────────────────────


def test_the_summary_row_and_every_pass_name_the_serving_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _samples(6)
    _wire(monkeypatch, _LazyLoader(samples), _adapter(monkeypatch, _Server([BLOCK_A])))
    _run(tmp_path, concurrency=3)
    assert _row(tmp_path)["serving"] == BLOCK_A
    assert [p["serving"] for p in _passes(tmp_path)] == [BLOCK_A]
    assert serving.serving_problem(_row(tmp_path)) == ""


def test_a_server_without_a_block_records_null_and_warns_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    samples = _samples(4)
    _wire(monkeypatch, _LazyLoader(samples), _adapter(monkeypatch, _Server([None])))
    with caplog.at_level("WARNING"):
        _run(tmp_path)
    assert _row(tmp_path)["serving"] is None
    assert _passes(tmp_path)[0]["serving"] is None
    assert sum("serving recipe" in r.getMessage() for r in caplog.records) == 1


def test_hosted_adapters_record_null_without_a_pass_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_resume import _NoisyAdapter

    samples = _samples(3)
    _wire(monkeypatch, _LazyLoader(samples), _NoisyAdapter(samples))
    _run(tmp_path)
    assert _row(tmp_path)["serving"] is None
    assert "serving" not in _passes(tmp_path)[0]


# ── A change of setup aborts ─────────────────────────────────────────────────


def test_a_block_that_changes_mid_run_aborts_and_records_nothing_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _samples(6)
    _wire(monkeypatch, _LazyLoader(samples),
          _adapter(monkeypatch, _Server([BLOCK_A, BLOCK_A, BLOCK_B])))
    with pytest.raises(serving.ServingChangedError, match="changed mid-run"):
        _run(tmp_path, concurrency=1)
    lines = (tmp_path / f"predictions_{SUBSET}.jsonl").read_text().splitlines()
    assert len(lines) == 2, "the answer from the changed setup must not be recorded"
    assert not (tmp_path / f".done_{SUBSET}.json").exists()
    # The interrupted pass still names the setup its transcripts came from.
    assert _passes(tmp_path)[0]["serving"] == BLOCK_A


def test_reproduce_exits_3_on_a_serving_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _samples(4)
    _wire(monkeypatch, _LazyLoader(samples),
          _adapter(monkeypatch, _Server([BLOCK_A, BLOCK_B])))
    rc = reproduce.main(["--metric", "wer", "--dataset", SUBSET, "--model", MODEL,
                         "--concurrency", "1", "--out", str(tmp_path)])
    assert rc == reproduce.EXIT_SERVED_MODEL_MISMATCH == 3


def test_resuming_under_another_setup_is_refused_at_its_first_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _samples(6)

    class _FailSome(_Server):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            if self.n == 2:
                self.n += 1
                return httpx.Response(400, json={"error": "bad clip"})
            return super().__call__(request)

    _wire(monkeypatch, _LazyLoader(samples), _adapter(monkeypatch, _FailSome([BLOCK_A])))
    _run(tmp_path)  # incomplete: one failed clip, the subset stays resumable
    assert not (tmp_path / f".done_{SUBSET}.json").exists()

    second = _Server([BLOCK_B])
    _wire(monkeypatch, _LazyLoader(samples), _adapter(monkeypatch, second))
    before = (tmp_path / f"predictions_{SUBSET}.jsonl").read_text()
    with pytest.raises(serving.ServingChangedError, match="earlier passes"):
        _run(tmp_path)
    assert second.n == 1, "the run went on after the first answer from another setup"
    assert (tmp_path / f"predictions_{SUBSET}.jsonl").read_text() == before

    # The setup the earlier passes recorded resumes normally.
    _wire(monkeypatch, _LazyLoader(samples), _adapter(monkeypatch, _Server([BLOCK_A])))
    _run(tmp_path)
    assert (tmp_path / f".done_{SUBSET}.json").exists()
    assert _row(tmp_path)["serving"] == BLOCK_A


def test_disagreeing_passes_have_no_single_block() -> None:
    with pytest.raises(serving.ServingChangedError):
        serving.of_passes([{"serving": BLOCK_A}, {"serving": BLOCK_B}])
    # Lines written before serving was recorded do not count.
    assert serving.of_passes([{"wall_s": 1}, {"serving": BLOCK_A}]) == BLOCK_A
    assert serving.of_passes([{"wall_s": 1}]) is None


# ── promote / verify ─────────────────────────────────────────────────────────


def test_verify_holds_the_block_to_its_passes() -> None:
    row = {"serving": BLOCK_A, "throughput": {"passes": [{"serving": BLOCK_A}]}}
    assert serving.serving_problem(row) == ""
    assert "vs passes" in serving.serving_problem(
        {**row, "serving": BLOCK_B}
    )
    assert "different setups" in serving.serving_problem(
        {"serving": BLOCK_A, "throughput": {"passes": [{"serving": BLOCK_A}, {"serving": BLOCK_B}]}}
    )
    assert "recipe" in serving.serving_problem(
        {"serving": {**BLOCK_A, "recipe": {"id": "x", "sha256": "short"}}}
    )
    assert "keys" in serving.serving_problem({"serving": {"engine": "vllm"}})
    # Artifacts from before serving was recorded pass.
    assert serving.serving_problem({"throughput": {"passes": [{"wall_s": 1}]}}) == ""
    assert serving.serving_problem({"serving": None}) == ""


def test_promote_carries_the_block_into_the_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _samples(4)
    run_dir = tmp_path / "run"
    _wire(monkeypatch, _LazyLoader(samples), _adapter(monkeypatch, _Server([BLOCK_A])))
    _run(run_dir)
    dest = promote_mod.promote(run_dir, tmp_path / "artifacts", "t")
    (row,) = json.loads((dest / "summary.json").read_text())["results"]
    assert row["serving"] == BLOCK_A
    tp = [json.loads(x) for x in (dest / f"throughput_{SUBSET}.jsonl").read_text().splitlines()]
    assert tp[0]["serving"] == BLOCK_A


def test_a_block_recorded_before_decoding_was_reported_is_still_well_formed() -> None:
    legacy = {k: v for k, v in BLOCK_A.items() if k not in ("decoding", "precision")}
    assert serving.serving_problem({"serving": legacy}) == ""
    assert serving.serving_problem({"serving": {**BLOCK_A, "decoding": "greedy"}}) != ""


def test_a_change_of_output_cap_is_a_change_of_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = {**BLOCK_A, "decoding": {"mode": "greedy", "max_tokens": 4096}}
    samples = _samples(4)
    _wire(monkeypatch, _LazyLoader(samples), _adapter(monkeypatch, _Server([BLOCK_A, other])))
    with pytest.raises(serving.ServingChangedError, match="changed mid-run"):
        _run(tmp_path)


# ── Precision (issue #81) ────────────────────────────────────────────────────

_W8A16 = {
    "weights": "fp8-e4m3", "activations": "bf16", "scheme": "fp8-w8a16-marlin",
    "tensor_parallel": 2,
    "checkpoint": {"source": "vendor", "repo": "org/m-fp8", "revision": "b" * 40,
                   "dtype": "fp8-e4m3"},
}


def test_a_change_of_precision_mid_subset_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restart of the same recipe id on quantized weights is another setup."""
    other = {**BLOCK_A, "precision": _W8A16}
    samples = _samples(4)
    _wire(monkeypatch, _LazyLoader(samples), _adapter(monkeypatch, _Server([BLOCK_A, other])))
    with pytest.raises(serving.ServingChangedError, match="changed mid-run"):
        _run(tmp_path)


def test_a_server_that_names_its_recipe_but_not_its_precision_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Serving code from before #81: its results could not name their precision."""
    stale = {k: v for k, v in BLOCK_A.items() if k != "precision"}
    samples = _samples(3)
    _wire(monkeypatch, _LazyLoader(samples), _adapter(monkeypatch, _Server([stale])))
    with pytest.raises(serving.ServingChangedError, match="not its precision"):
        _run(tmp_path)
    assert not (tmp_path / f"throughput_{SUBSET}.jsonl").exists() or not _passes(tmp_path)


def test_a_block_recorded_before_precision_was_reported_is_still_well_formed() -> None:
    pre = {k: v for k, v in BLOCK_A.items() if k != "precision"}
    assert serving.serving_problem({"serving": pre}) == ""
    assert serving.serving_problem({"serving": {**BLOCK_A, "precision": None}}) == ""
    assert serving.serving_problem({"serving": {**BLOCK_A, "precision": _W8A16}}) == ""


@pytest.mark.parametrize(
    "bad",
    [
        "bf16",
        {k: v for k, v in _W8A16.items() if k != "tensor_parallel"},
        {**_W8A16, "tensor_parallel": 0},
        {**_W8A16, "weights": ""},
        {**_W8A16, "checkpoint": {"source": "vendor"}},
    ],
)
def test_a_malformed_precision_fails_the_artifact(bad: object) -> None:
    assert "serving.precision" in serving.serving_problem({"serving": {**BLOCK_A, "precision": bad}})


def test_the_published_label_names_scheme_cast_and_split() -> None:
    assert serving.precision_label(BLOCK_A["precision"]) == "fp16 (cast from bf16)"
    assert serving.precision_label(_W8A16) == "fp8-e4m3 fp8-w8a16-marlin, TP 2"
    plain = {**BLOCK_A["precision"], "weights": "bf16", "activations": "bf16",
             "checkpoint": {"source": "published", "dtype": "bf16"}}
    assert serving.precision_label(plain) == "bf16"
