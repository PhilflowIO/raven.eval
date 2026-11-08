"""Build the int8 weight-only (W8A16) checkpoint of Nemotron-3.5-Transcribe (#78).

    python serving/quantize/nemotron_transcribe_w8a16.py all --work <dir> --out <dir>
    python serving/quantize/nemotron_transcribe_w8a16.py {download,convert,quantize,verify,sums} ...

What it makes. NVIDIA's bf16 release (REPO at REVISION, one 67.5 GB
``model.safetensors``) with every Linear weight of the language model's decoder
layers rounded to int8, one scale per output channel, symmetric, round to
nearest (no calibration data). The audio encoder ("perception"), its projector,
the bundled CTC timestamp head, the MTP draft layers, embeddings, lm_head, the
MoE routers, Mamba conv1d and every norm stay bit-identical bf16/fp32. vLLM
0.33.0 serves it as compressed-tensors W8A16 (Marlin on an RTX 3090) with
bf16 activations, split over two cards; the precision table of its recipe names
this script as ``checkpoint.recipe``.

Why not NVIDIA's FP8 build: its audio encoder is FP8 too, and the NeMo plugin's
FP8 encoder layer calls a CUTLASS FP8 GEMM that has no sm86 kernel
(nemo/collections/speechlm2/vllm/salm/fp8_encoder.py, ``cutlass_scaled_mm``).

Steps (``all`` runs them in order; each one can be re-run on its own):

  download  the release at REVISION into the HF cache (or pass --source <dir>).
  convert   stream the NeMo-layout tensors into the Hugging Face NemotronH layout
            the plugin itself turns them into before vLLM sees them (backends.py
            HybridBackend.nemo_to_hf_llm_weights at the pinned plugin commit):
            ``llm.model.`` -> ``backbone.``, packed experts [E, in, out] ->
            per-expert ``experts.{i}.{up,down}_proj.weight`` [out, in], the Mamba
            fp32 holders -> ``mixer.{A,D,dt_bias}``, ``backbone.norm`` ->
            ``backbone.norm_f``, ``._extra_state`` dropped. Bytes are moved, never
            re-encoded: every output tensor is a byte-for-byte copy or transpose of
            a source tensor. Shards of <= 5 GiB, so the quantizer holds little in
            RAM; the CTC head goes to its own file with the source's metadata
            (the plugin reads the head's configuration from there), and so do the
            MTP draft layers, whose fused experts the quantizer would otherwise
            rewrite (see restore_passthrough).
  quantize  llm-compressor model_free_ptq over the converted shards on the CPU
            (QUANT_TARGETS / QUANT_IGNORE below), then the CTC and MTP files are
            restored byte-identical (convert_checkpoint rewrites every shard
            without its safetensors metadata, and splits 3-D tensors whose name
            contains "down_proj", which the MTP layer's packed experts do).
  verify    the output against the source: the scheme in config.json, exactly
            the expected Linear set quantized (EXPECTED_QUANTIZED), every other
            tensor byte-identical to the converted input.
  sums      SHA256SUMS over every output file and the sha256 of that list (the
            checkpoint hash a recipe records).

Host: CPU only, no GPU. Peak RAM is about --workers x 2 x 5 GiB plus the page
cache; disk: the 67.5 GB source, ~67 GB converted, ~36 GB output.

Environment: serving/quantize/requirements.txt (llm-compressor 0.15.0,
compressed-tensors 0.20.1, torch 2.14.1). ``convert``, ``verify`` and ``sums``
need only numpy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import re
import shutil
import struct
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
REVISION = "c17667641642c4265ff2b7037dd101b7d4567634"
SOURCE_FILE = "model.safetensors"
#: The NeMo plugin commit whose weight renames ``convert`` reproduces
#: (serving/vllm/requirements.txt pins the same one).
PLUGIN_COMMIT = "821e036741a60332ca90b160243d88d13a90ec84"

SHARD_BYTES = 5 * 2**30
CTC_FILE = "model-ctc-bf16.safetensors"
CTC_PREFIX = "ctc_timestamp."
MTP_FILE = "model-mtp-bf16.safetensors"
MTP_PREFIX = "llm.mtp."
INDEX_FILE = "model.safetensors.index.json"
SUMS_FILE = "SHA256SUMS"

#: Everything in the release except its single weights file is copied as is.
COPY_EXCLUDE = frozenset({SOURCE_FILE, ".gitattributes"})

#: Mamba fp32 parameter holders -> the names vLLM's NemotronH loads
#: (backends.py _FP32_PARAM_HOLDER_RENAMES at PLUGIN_COMMIT).
FP32_HOLDER_RENAMES = {
    ".mixer._fp32_params.A_log": ".mixer.A",
    ".mixer._fp32_params.dt_bias": ".mixer.dt_bias",
    ".mixer._fp32_params.D": ".mixer.D",
}
_PACKED_EXPERTS = {".experts.gate_and_up_projs": "up_proj", ".experts.down_projs": "down_proj"}

#: Quantized: every Linear of the decoder layers. "Linear" plus an ignore list is
#: the form vLLM's compressed-tensors MoE path needs (it looks the experts up
#: under the name "<layer>.0.None" too, which only a class target matches); the
#: same ignore list keeps model_free_ptq, which treats every 2-D ``*.weight`` as
#: a Linear, off everything else. Patterns match the converted names
#: (``backbone.layers.3.mixer.in_proj``) and vLLM's
#: (``language_model.model.layers.3.mixer.in_proj``) alike.
QUANT_TARGETS = ["Linear"]
QUANT_IGNORE = [
    r"re:^perception\..*",  # audio encoder + projector: bf16 (sm86 has no FP8 path)
    r"re:^ctc_timestamp\..*",  # CTC timestamp head
    r"re:^llm\.mtp\..*",  # MTP draft layers (not served)
    r"re:.*lm_head$",
    r"re:.*embed_tokens$",
    r"re:.*embeddings$",
    r"re:.*mixer\.gate$",  # MoE router
    r"re:.*mixer\.conv1d$",  # Mamba conv (vLLM builds it as a Linear)
    r"re:.*norm_f$",  # final norm (model_free_ptq skips only names ending in "norm")
]
#: int8, one scale per output channel, symmetric, static, weights only. NOT the
#: W8A16 preset: that one is group-128, and at tensor-parallel 2 an expert's
#: down_proj has 1856 / 2 = 928 input columns per card, which 128 does not
#: divide (vLLM refuses such a MoE layer).
WEIGHT_ARGS = {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel",
               "dynamic": False}

#: The decoder layout of the release (llm_backbone/config.json): 52 layers =
#: 23 Mamba (in_proj, out_proj) + 6 attention (q, k, v, o) + 23 MoE (128 routed
#: experts with up/down + one shared expert with up/down).
N_MAMBA, N_ATTN, N_MOE, N_EXPERTS = 23, 6, 23, 128
EXPECTED_QUANTIZED = N_MAMBA * 2 + N_ATTN * 4 + N_MOE * (N_EXPERTS * 2 + 2)

_ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2,
             "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}
_NP_WORD = {1: "u1", 2: "<u2", 4: "<u4", 8: "<u8"}


# ── safetensors, read and written as bytes ──────────────────────────────────


def read_header(path: Path) -> tuple[dict[str, Any], dict[str, str], int]:
    """``(tensors, metadata, data_start)`` of a safetensors file."""
    with path.open("rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
    metadata = header.pop("__metadata__", None) or {}
    return header, metadata, 8 + n


def _nbytes(dtype: str, shape: list[int]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count * _ITEMSIZE[dtype]


def write_safetensors(path: Path, tensors: list[tuple[str, str, list[int], Any]],
                      metadata: dict[str, str] | None = None) -> None:
    """Write ``[(name, dtype, shape, producer)]``; ``producer()`` returns the bytes.

    Header layout as the reference implementation writes it (JSON padded with
    spaces to a multiple of 8), tensors in the given order, back to back.
    """
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    for name, dtype, shape, _ in tensors:
        size = _nbytes(dtype, shape)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        for name, dtype, shape, produce in tensors:
            data = produce()
            if len(data) != _nbytes(dtype, shape):
                raise RuntimeError(f"{name}: produced {len(data)} bytes, expected {_nbytes(dtype, shape)}")
            fh.write(data)
    tmp.replace(path)


# ── convert ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Out:
    """One output tensor: ``source[expert]`` (transposed) or ``source`` as is."""

    name: str
    dtype: str
    shape: list[int]
    source: str
    expert: int | None = None


def hf_name(name: str) -> str:
    """The plugin's NeMo -> HF NemotronH rename for one non-expert tensor."""
    out = name.replace("llm.model.", "backbone.").replace("llm.lm_head", "lm_head")
    for holder, target in FP32_HOLDER_RENAMES.items():
        if out.endswith(holder):
            out = out[: -len(holder)] + target
    return "backbone.norm_f.weight" if out == "backbone.norm.weight" else out


def plan(header: dict[str, Any]) -> list[Out]:
    """Every output tensor of the converted checkpoint, in a fixed order."""
    outs: list[Out] = []
    for name in sorted(header):
        info = header[name]
        if "._extra_state" in name:
            continue
        if name.startswith("llm.mtp."):
            # The plugin hands draft tensors to the MTP model untouched; keep them so.
            outs.append(Out(name, info["dtype"], list(info["shape"]), name))
            continue
        packed = next((s for s in _PACKED_EXPERTS if name.endswith(s)), None)
        if packed is not None:
            n_exp, rows, cols = info["shape"]
            prefix = hf_name(name[: -len(packed)])
            for i in range(n_exp):
                outs.append(Out(f"{prefix}.experts.{i}.{_PACKED_EXPERTS[packed]}.weight",
                                info["dtype"], [cols, rows], name, i))
            continue
        outs.append(Out(hf_name(name), info["dtype"], list(info["shape"]), name))
    names = [o.name for o in outs]
    if len(set(names)) != len(names):
        raise RuntimeError("two source tensors map to one converted name")
    return outs


def _apart(name: str) -> str | None:
    """The file of its own a tensor goes to, or None for the numbered shards."""
    if name.startswith(CTC_PREFIX):
        return CTC_FILE
    if name.startswith(MTP_PREFIX):
        return MTP_FILE
    return None


def shards(outs: list[Out], limit: int | None = None) -> list[list[Out]]:
    """The CTC head and the MTP layers on their own; the rest in order, <= ``limit`` bytes per shard."""
    limit = limit or SHARD_BYTES
    ctc = [o for o in outs if _apart(o.name) == CTC_FILE]
    mtp = [o for o in outs if _apart(o.name) == MTP_FILE]
    groups: list[list[Out]] = [[]]
    size = 0
    for o in outs:
        if _apart(o.name) is not None:
            continue
        n = _nbytes(o.dtype, o.shape)
        if groups[-1] and size + n > limit:
            groups.append([])
            size = 0
        groups[-1].append(o)
        size += n
    return [g for g in groups if g] + ([mtp] if mtp else []) + ([ctc] if ctc else [])


def _producer(buf: mmap.mmap, start: int, info: dict[str, Any], out: Out):
    begin, end = (start + x for x in info["data_offsets"])
    if out.expert is None:
        return lambda: bytes(buf[begin:end])
    import numpy as np

    _, rows, cols = info["shape"]
    width = _ITEMSIZE[info["dtype"]]
    step = rows * cols * width
    lo = begin + out.expert * step

    def transposed() -> bytes:
        block = np.frombuffer(buf, dtype=_NP_WORD[width], count=rows * cols, offset=lo)
        return np.ascontiguousarray(block.reshape(rows, cols).T).tobytes()

    return transposed


def convert(source: Path, work: Path) -> None:
    weights = source / SOURCE_FILE
    header, metadata, start = read_header(weights)
    outs = plan(header)
    groups = shards(outs)
    work.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.rglob("*")):
        rel = item.relative_to(source)
        if item.is_file() and str(rel) not in COPY_EXCLUDE:
            (work / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, work / rel)
    weight_map: dict[str, str] = {}
    total = 0
    n_main = sum(1 for g in groups if _apart(g[0].name) is None)
    with weights.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        for i, group in enumerate(groups):
            apart = _apart(group[0].name)
            fname = apart or f"model-{i + 1:05d}-of-{n_main:05d}.safetensors"
            meta = metadata if apart == CTC_FILE else {"format": "pt"}
            write_safetensors(work / fname, [
                (o.name, o.dtype, o.shape, _producer(buf, start, header[o.source], o)) for o in group
            ], meta)
            for o in group:
                weight_map[o.name] = fname
                total += _nbytes(o.dtype, o.shape)
            print(f"[convert] {fname}: {len(group)} tensors", flush=True)
    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}
    (work / INDEX_FILE).write_text(json.dumps(index, indent=2) + "\n")


# ── quantize ────────────────────────────────────────────────────────────────


def quantize(work: Path, out: Path, workers: int) -> None:
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from llmcompressor import model_free_ptq

    scheme = QuantizationScheme(targets=QUANT_TARGETS, weights=QuantizationArgs(**WEIGHT_ARGS))
    model_free_ptq(model_stub=str(work), save_directory=str(out), scheme=scheme,
                   ignore=QUANT_IGNORE, max_workers=workers, device="cpu")
    restore_passthrough(work, out)


def restore_passthrough(work: Path, out: Path) -> None:
    """Put the CTC and MTP files back byte-identical, under the converted names.

    The CTC tensors must be proven unchanged first (only its metadata is lost).
    The MTP file cannot be compared name by name: ``model_free_ptq`` splits every
    3-D tensor whose name contains "down_proj", which the MTP layer's packed
    ``experts.down_projs`` does, and renames it. No MTP tensor is quantized
    (QUANT_IGNORE), so the file is the converted one; ``verify`` proves it and the
    index is put back for its tensors.
    """
    src, dst = work / CTC_FILE, out / CTC_FILE
    if src.exists():
        a, _, sa = read_header(src)
        b, _, sb = read_header(dst)
        if set(a) != set(b) or any(_tensor_bytes(src, sa, a[k]) != _tensor_bytes(dst, sb, b[k])
                                   or a[k]["dtype"] != b[k]["dtype"] for k in a):
            raise RuntimeError(f"{dst} differs from {src} beyond its metadata")
        shutil.copyfile(src, dst)
    src, dst = work / MTP_FILE, out / MTP_FILE
    if src.exists():
        shutil.copyfile(src, dst)
        converted = json.loads((work / INDEX_FILE).read_text())
        index = json.loads((out / INDEX_FILE).read_text())
        weight_map = {n: f for n, f in index["weight_map"].items() if f != MTP_FILE}
        weight_map |= {n: f for n, f in converted["weight_map"].items() if f == MTP_FILE}
        index["weight_map"] = dict(sorted(weight_map.items()))
        (out / INDEX_FILE).write_text(json.dumps(index, indent=2) + "\n")


# ── verify / sums ───────────────────────────────────────────────────────────


def _tensor_bytes(path: Path, start: int, info: dict[str, Any]) -> bytes:
    begin, end = info["data_offsets"]
    with path.open("rb") as fh:
        fh.seek(start + begin)
        return fh.read(end - begin)


def _tensors(root: Path) -> Iterator[tuple[str, Path, int, dict[str, Any]]]:
    weight_map = json.loads((root / INDEX_FILE).read_text())["weight_map"]
    headers: dict[str, tuple[dict[str, Any], int]] = {}
    for name, fname in sorted(weight_map.items()):
        if fname not in headers:
            h, _, s = read_header(root / fname)
            headers[fname] = (h, s)
        h, s = headers[fname]
        yield name, root / fname, s, h[name]


def verify(work: Path, out: Path) -> None:
    config = json.loads((out / "config.json").read_text())
    qc = config.get("quantization_config") or {}
    groups = list((qc.get("config_groups") or {}).values())
    if qc.get("quant_method") != "compressed-tensors" or len(groups) != 1:
        raise SystemExit(f"config.json: expected one compressed-tensors group, got {qc!r:.300}")
    weights = groups[0].get("weights") or {}
    for key, value in WEIGHT_ARGS.items():
        if weights.get(key) != value:
            raise SystemExit(f"config.json: weights.{key} = {weights.get(key)!r}, expected {value!r}")
    if groups[0].get("input_activations") is not None:
        raise SystemExit("config.json: activations are quantized; W8A16 keeps them 16-bit")
    if qc.get("format") != "pack-quantized":
        raise SystemExit(f"config.json: format {qc.get('format')!r}; vLLM's WNA16 MoE needs pack-quantized")

    before = {name: (p, s, info) for name, p, s, info in _tensors(work)}
    after = {name: (p, s, info) for name, p, s, info in _tensors(out)}
    quantized = sorted({n.rsplit(".", 1)[0] for n in after if n.endswith(".weight_packed")})
    if len(quantized) != EXPECTED_QUANTIZED:
        raise SystemExit(f"{len(quantized)} Linears quantized, expected {EXPECTED_QUANTIZED}")
    pattern = re.compile(r"^backbone\.layers\.\d+\.mixer\.(in_proj|out_proj|[qkvo]_proj|"
                         r"experts\.\d+\.(up|down)_proj|shared_experts\.(up|down)_proj)$")
    if stray := [m for m in quantized if not pattern.match(m)]:
        raise SystemExit(f"quantized outside the decoder Linears: {stray[:5]}")
    for module in quantized:
        if f"{module}.weight" in after or f"{module}.weight_scale" not in after:
            raise SystemExit(f"{module}: expected weight_packed + weight_scale and no bf16 weight")
        if f"{module}.weight" not in before:
            raise SystemExit(f"{module}: not in the converted input")
    qset = set(quantized)
    kept = [n for n in before if n.rsplit(".", 1)[0] not in qset]
    for name in kept:
        if name not in after:
            raise SystemExit(f"{name} is missing from the output")
        a, b = before[name], after[name]
        if a[2]["dtype"] != b[2]["dtype"] or a[2]["shape"] != b[2]["shape"] or \
                _tensor_bytes(*a) != _tensor_bytes(*b):
            raise SystemExit(f"{name} changed although it is not quantized")
    print(f"[verify] ok: {len(quantized)} Linears int8 per-channel, {len(kept)} tensors unchanged")


def sums(out: Path) -> str:
    """Write SHA256SUMS (sha256sum format) and return its own sha256."""
    lines = []
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != SUMS_FILE):
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            while chunk := fh.read(1 << 24):
                digest.update(chunk)
        lines.append(f"{digest.hexdigest()}  {path.relative_to(out).as_posix()}\n")
    text = "".join(lines)
    (out / SUMS_FILE).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def download() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(REPO, revision=REVISION))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("step", choices=["download", "convert", "quantize", "verify", "sums", "all"])
    ap.add_argument("--source", type=Path, help="the release snapshot (default: download it)")
    ap.add_argument("--work", type=Path, help="converted, unquantized checkpoint")
    ap.add_argument("--out", type=Path, help="the quantized checkpoint")
    ap.add_argument("--workers", type=int, default=4, help="shards quantized at once (RAM!)")
    args = ap.parse_args(argv)
    needs = {"convert": ["work"], "quantize": ["work", "out"], "verify": ["work", "out"],
             "sums": ["out"], "all": ["work", "out"]}.get(args.step, [])
    if missing := [f"--{n}" for n in needs if getattr(args, n) is None]:
        ap.error(f"{args.step} needs {' '.join(missing)}")
    if args.step in ("download", "convert", "all") and args.source is None:
        args.source = download()
        print(f"[download] {REPO}@{REVISION}: {args.source}", flush=True)
    if args.step in ("convert", "all"):
        convert(args.source, args.work)
    if args.step in ("quantize", "all"):
        quantize(args.work, args.out, args.workers)
    if args.step in ("verify", "all"):
        verify(args.work, args.out)
    if args.step in ("sums", "all"):
        print(f"[sums] {args.out / SUMS_FILE}: sha256 {sums(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
