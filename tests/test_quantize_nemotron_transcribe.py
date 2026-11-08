"""The conversion step of serving/quantize/nemotron_transcribe_w8a16.py (#78).

It turns NVIDIA's NeMo-layout release into the layout NeMo's vLLM plugin hands
to vLLM, by moving bytes only. These tests build a tiny release with the same
naming and check names, shapes, the expert transposes byte for byte, the CTC
file with its metadata, and that the files are valid safetensors.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "serving" / "quantize" / "nemotron_transcribe_w8a16.py"


@pytest.fixture(scope="module")
def q():
    spec = importlib.util.spec_from_file_location("nemotron_transcribe_w8a16", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bf16(rng, *shape):
    return rng.integers(0, 2**16, size=shape, dtype=np.uint16)


@pytest.fixture()
def release(tmp_path, q):
    """A two-expert, one-layer release in the NeMo layout, plus its side files."""
    rng = np.random.default_rng(0)
    src = tmp_path / "release"
    (src / "llm_backbone").mkdir(parents=True)
    arrays = {
        "llm.model.layers.0.mixer.experts.gate_and_up_projs": ("BF16", _bf16(rng, 2, 4, 3)),
        "llm.model.layers.0.mixer.experts.down_projs": ("BF16", _bf16(rng, 2, 3, 4)),
        "llm.model.layers.0.mixer.shared_experts.up_proj.weight": ("BF16", _bf16(rng, 6, 4)),
        "llm.model.layers.0.mixer.shared_experts.up_proj._extra_state": ("BF16", _bf16(rng, 0)),
        "llm.model.layers.1.mixer._fp32_params.A_log": ("F32", rng.random(2, dtype=np.float32)),
        "llm.model.layers.1.mixer.in_proj.weight": ("BF16", _bf16(rng, 5, 4)),
        "llm.model.norm.weight": ("BF16", _bf16(rng, 4)),
        "llm.model.embed_tokens.weight": ("BF16", _bf16(rng, 8, 4)),
        "llm.lm_head.weight": ("BF16", _bf16(rng, 8, 4)),
        "llm.mtp.layers.0.mixer.experts.down_projs": ("BF16", _bf16(rng, 2, 3, 4)),
        "perception.encoder.layers.0.linear.weight": ("BF16", _bf16(rng, 4, 4)),
        "ctc_timestamp.decoder.upsample.bias": ("F32", rng.random(4, dtype=np.float32)),
        "ctc_timestamp.tokenizer.model_proto": ("U8", rng.integers(0, 255, 7, dtype=np.uint8)),
    }
    meta = {"ctc_timestamp_format": "nemo_ctc_timestamp_artifact_v1", "ctc_timestamp_decoder_config": "{}"}
    q.write_safetensors(src / q.SOURCE_FILE, [
        (name, dtype, list(a.shape), (lambda a=a: a.tobytes())) for name, (dtype, a) in arrays.items()
    ], meta)
    (src / "config.json").write_text('{"model_type": "nemo_speechlm"}')
    (src / "llm_backbone" / "config.json").write_text("{}")
    (src / ".gitattributes").write_text("*")
    return src, arrays, meta


def _read(q, path: Path, name: str) -> bytes:
    header, _, start = q.read_header(path)
    begin, end = header[name]["data_offsets"]
    return path.read_bytes()[start + begin:start + end]


def test_convert_moves_bytes_into_the_plugins_layout(q, release, tmp_path):
    src, arrays, meta = release
    work = tmp_path / "work"
    q.convert(src, work)
    index = json.loads((work / q.INDEX_FILE).read_text())["weight_map"]

    # Experts: one [out, in] tensor per expert, the transpose of its slice.
    for packed, proj in (("gate_and_up_projs", "up_proj"), ("down_projs", "down_proj")):
        _, stacked = arrays[f"llm.model.layers.0.mixer.experts.{packed}"]
        for i in range(2):
            name = f"backbone.layers.0.mixer.experts.{i}.{proj}.weight"
            header, _, _ = q.read_header(work / index[name])
            assert header[name]["shape"] == [stacked.shape[2], stacked.shape[1]]
            assert _read(q, work / index[name], name) == np.ascontiguousarray(stacked[i].T).tobytes()

    renamed = {
        "llm.model.layers.0.mixer.shared_experts.up_proj.weight":
            "backbone.layers.0.mixer.shared_experts.up_proj.weight",
        "llm.model.layers.1.mixer._fp32_params.A_log": "backbone.layers.1.mixer.A",
        "llm.model.layers.1.mixer.in_proj.weight": "backbone.layers.1.mixer.in_proj.weight",
        "llm.model.norm.weight": "backbone.norm_f.weight",
        "llm.model.embed_tokens.weight": "backbone.embed_tokens.weight",
        "llm.lm_head.weight": "lm_head.weight",
        # Untouched: the plugin passes these through under their NeMo names.
        "llm.mtp.layers.0.mixer.experts.down_projs": "llm.mtp.layers.0.mixer.experts.down_projs",
        "perception.encoder.layers.0.linear.weight": "perception.encoder.layers.0.linear.weight",
    }
    for old, new in renamed.items():
        assert _read(q, work / index[new], new) == arrays[old][1].tobytes(), new
    assert not any("_extra_state" in n or n.startswith("llm.model.") for n in index)

    # The CTC head sits in its own file, with the metadata the plugin reads it by.
    ctc = [n for n in index if n.startswith("ctc_timestamp.")]
    assert {index[n] for n in ctc} == {q.CTC_FILE}
    assert q.read_header(work / q.CTC_FILE)[1] == meta
    assert all(index[n] != q.CTC_FILE for n in index if n not in ctc)

    # Side files copied, the single weights file and .gitattributes not.
    assert (work / "llm_backbone" / "config.json").exists()
    assert not (work / q.SOURCE_FILE).exists() and not (work / ".gitattributes").exists()


def test_converted_files_are_valid_safetensors(q, release, tmp_path):
    safetensors = pytest.importorskip("safetensors")
    src, _, meta = release
    work = tmp_path / "work"
    q.convert(src, work)
    index = json.loads((work / q.INDEX_FILE).read_text())["weight_map"]
    for fname in sorted(set(index.values())):
        with safetensors.safe_open(str(work / fname), framework="numpy") as fh:
            assert set(fh.keys()) == {n for n, f in index.items() if f == fname}
            if fname == q.CTC_FILE:
                assert fh.metadata() == meta
                assert fh.get_tensor("ctc_timestamp.decoder.upsample.bias").dtype == np.float32


def test_shards_respect_the_limit_and_keep_the_ctc_head_apart(q, release):
    src, _, _ = release
    header, _, _ = q.read_header(src / q.SOURCE_FILE)
    outs = q.plan(header)
    groups = q.shards(outs, limit=64)
    assert groups[-1] and all(o.name.startswith("ctc_timestamp.") for o in groups[-1])
    assert groups[-2] and all(o.name.startswith("llm.mtp.") for o in groups[-2])
    for group in groups[:-2]:
        assert not any(o.name.startswith("ctc_timestamp.") for o in group)
        assert len(group) == 1 or sum(q._nbytes(o.dtype, o.shape) for o in group) <= 64
    assert sorted(o.name for g in groups for o in g) == sorted(o.name for o in outs)


def test_mtp_layers_get_their_own_file(q, release, tmp_path):
    src, _, _ = release
    work = tmp_path / "work"
    q.convert(src, work)
    index = json.loads((work / q.INDEX_FILE).read_text())["weight_map"]
    mtp = [n for n in index if n.startswith("llm.mtp.")]
    assert mtp and {index[n] for n in mtp} == {q.MTP_FILE}
    assert all(index[n] != q.MTP_FILE for n in index if n not in mtp)
    numbered = sorted(f for f in set(index.values()) if f not in (q.CTC_FILE, q.MTP_FILE))
    assert numbered == [f"model-{i + 1:05d}-of-{len(numbered):05d}.safetensors" for i in range(len(numbered))]


def test_restore_puts_the_split_mtp_tensors_back(q, release, tmp_path):
    """model_free_ptq splits any 3-D tensor named *down_proj* and renames it: the
    MTP layer's packed experts must come back byte-identical, under their names."""
    src, _, _ = release
    work, out = tmp_path / "work", tmp_path / "out"
    q.convert(src, work)
    shutil.copytree(work, out)
    # What the quantizer does: the file holds other names, the index follows it.
    original = (work / q.MTP_FILE).read_bytes()
    (out / q.MTP_FILE).write_bytes(b"rewritten by the quantizer")
    index = json.loads((out / q.INDEX_FILE).read_text())
    split_name = "llm.mtp.layers.0.mixer.experts.down_projs"
    assert index["weight_map"][split_name] == q.MTP_FILE
    del index["weight_map"][split_name]
    index["weight_map"]["llm.mtp.layers.0.mixer.experts.0.down_projs.weight"] = q.MTP_FILE
    (out / q.INDEX_FILE).write_text(json.dumps(index))

    q.restore_passthrough(work, out)

    assert (out / q.MTP_FILE).read_bytes() == original
    restored = json.loads((out / q.INDEX_FILE).read_text())["weight_map"]
    assert restored == json.loads((work / q.INDEX_FILE).read_text())["weight_map"]


def test_the_scheme_is_per_channel_int8_weight_only(q):
    """Group-128 (the W8A16 preset) would not divide an expert's 928 input columns
    per card at tensor-parallel 2."""
    assert q.WEIGHT_ARGS == {"num_bits": 8, "type": "int", "symmetric": True,
                             "strategy": "channel", "dynamic": False}
    assert q.QUANT_TARGETS == ["Linear"]
    assert q.EXPECTED_QUANTIZED == 23 * 2 + 6 * 4 + 23 * (128 * 2 + 2)
