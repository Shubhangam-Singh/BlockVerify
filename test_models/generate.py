#!/usr/bin/env python3
"""Build the BlockVerify test-model set: 7 model families plus their variants.

    python3 test_models/generate.py          (standard library only, seeded)

Every family has an original you register, and variants that are legitimate updates,
tampering attacks or harmless re-saves. The weights are synthetic (seeded Gaussian
draws at realistic scales). They are not trained models, and the metadata says so.

Writes the family folders, batch_scan/, MANIFEST.csv (file, kind, SHA-256, expected
result) and README.md next to this script. Re-running it gives byte-identical files.
"""
import csv
import hashlib
import json
import math
import os
import random
import shutil
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
NOTE = "Synthetic weights for BlockVerify testing. Not a trained model."


# ── architectures: (layer name, number of values, init) ─────────────────────────
# init: ("he", fan_in) | ("bias",) | ("norm_w",) | ("norm_b",) | ("embed", std) | ("empty",)
FAMILIES = {
    "cardio-ecg-cnn": {
        "meta": {"task": "ECG arrhythmia classification (5 classes)", "architecture": "1D CNN",
                 "framework": "PyTorch 2.4, exported to JSON", "input": "360-sample single-lead ECG window"},
        "layers": [
            ("conv1d_1.weight", 16 * 1 * 7, ("he", 7)), ("conv1d_1.bias", 16, ("bias",)),
            ("groupnorm_1.weight", 16, ("norm_w",)), ("groupnorm_1.bias", 16, ("norm_b",)),
            ("conv1d_2.weight", 32 * 16 * 5, ("he", 80)), ("conv1d_2.bias", 32, ("bias",)),
            ("groupnorm_2.weight", 32, ("norm_w",)), ("groupnorm_2.bias", 32, ("norm_b",)),
            ("global_avg_pool", 0, ("empty",)),
            ("dense_1.weight", 64 * 32, ("he", 32)), ("dense_1.bias", 64, ("bias",)),
            ("dense_out.weight", 5 * 64, ("he", 64)), ("dense_out.bias", 5, ("bias",)),
        ],
    },
    "retina-dr-resnet": {
        "meta": {"task": "Diabetic retinopathy grading (5 grades)", "architecture": "ResNet-style CNN",
                 "framework": "PyTorch 2.4, exported to JSON", "input": "224x224 RGB fundus image"},
        "layers": [
            ("stem.conv.weight", 16 * 3 * 3 * 3, ("he", 27)),
            ("stem.gn.weight", 16, ("norm_w",)), ("stem.gn.bias", 16, ("norm_b",)),
            ("block1.conv1.weight", 16 * 16 * 3 * 3, ("he", 144)),
            ("block1.gn1.weight", 16, ("norm_w",)), ("block1.gn1.bias", 16, ("norm_b",)),
            ("block1.conv2.weight", 16 * 16 * 3 * 3, ("he", 144)),
            ("block1.gn2.weight", 16, ("norm_w",)), ("block1.gn2.bias", 16, ("norm_b",)),
            ("block2.proj.weight", 32 * 16, ("he", 16)),
            ("block2.gn.weight", 32, ("norm_w",)), ("block2.gn.bias", 32, ("norm_b",)),
            ("head.fc.weight", 5 * 32, ("he", 32)), ("head.fc.bias", 5, ("bias",)),
        ],
    },
    "fraud-detector-mlp": {
        "meta": {"task": "Card-transaction fraud detection (binary)", "architecture": "MLP",
                 "framework": "PyTorch 2.4, exported to JSON", "input": "30 transaction features"},
        "layers": [
            ("input_norm.mean", 30, ("norm_b",)), ("input_norm.std", 30, ("norm_w",)),
            ("fc1.weight", 64 * 30, ("he", 30)), ("fc1.bias", 64, ("bias",)),
            ("fc2.weight", 32 * 64, ("he", 64)), ("fc2.bias", 32, ("bias",)),
            ("fc_out.weight", 32, ("he", 32)), ("fc_out.bias", 1, ("bias",)),
        ],
    },
    "sentiment-lstm": {
        "meta": {"task": "Review sentiment (negative / neutral / positive)", "architecture": "Embedding + LSTM",
                 "framework": "TensorFlow 2.17, exported to JSON", "input": "up to 64 tokens, 200-word vocabulary"},
        "layers": [
            ("embedding.weight", 200 * 16, ("embed", 0.1)),
            ("lstm.weight_ih", 64 * 16, ("he", 16)), ("lstm.weight_hh", 64 * 16, ("he", 16)),
            ("lstm.bias", 64, ("bias",)),
            ("dropout", 0, ("empty",)),
            ("classifier.weight", 3 * 16, ("he", 16)), ("classifier.bias", 3, ("bias",)),
        ],
    },
    "traffic-sign-vit": {
        "meta": {"task": "Traffic-sign recognition (43 classes)", "architecture": "Vision Transformer (1 block)",
                 "framework": "JAX/Flax, exported to JSON", "input": "16x16 RGB, 4x4 patches"},
        "layers": [
            ("patch_embed.proj.weight", 32 * 48, ("he", 48)), ("patch_embed.proj.bias", 32, ("bias",)),
            ("cls_token", 32, ("embed", 0.02)), ("pos_embed", 17 * 32, ("embed", 0.02)),
            ("block0.ln1.weight", 32, ("norm_w",)), ("block0.ln1.bias", 32, ("norm_b",)),
            ("block0.attn.qkv.weight", 96 * 32, ("he", 32)), ("block0.attn.qkv.bias", 96, ("bias",)),
            ("block0.attn.proj.weight", 32 * 32, ("he", 32)),
            ("block0.ln2.weight", 32, ("norm_w",)), ("block0.ln2.bias", 32, ("norm_b",)),
            ("block0.mlp.fc1.weight", 64 * 32, ("he", 32)), ("block0.mlp.fc2.weight", 32 * 64, ("he", 64)),
            ("head.weight", 43 * 32, ("he", 32)), ("head.bias", 43, ("bias",)),
        ],
    },
    "crop-disease-mobilenet": {
        "meta": {"task": "Plant-leaf disease classification (38 classes)", "architecture": "MobileNet-style CNN",
                 "framework": "TensorFlow 2.17, exported to JSON", "input": "128x128 RGB leaf image"},
        "layers": [
            ("conv_stem.weight", 16 * 27, ("he", 27)),
            ("gn_stem.weight", 16, ("norm_w",)), ("gn_stem.bias", 16, ("norm_b",)),
            ("dw1.weight", 16 * 9, ("he", 9)), ("pw1.weight", 32 * 16, ("he", 16)),
            ("gn1.weight", 32, ("norm_w",)), ("gn1.bias", 32, ("norm_b",)),
            ("dw2.weight", 32 * 9, ("he", 9)), ("pw2.weight", 64 * 32, ("he", 32)),
            ("gn2.weight", 64, ("norm_w",)), ("gn2.bias", 64, ("norm_b",)),
            ("classifier.weight", 38 * 64, ("he", 64)), ("classifier.bias", 38, ("bias",)),
        ],
    },
    "weather-forecast-gru": {          # never registered: the "unregistered" case
        "meta": {"task": "Next-hour temperature forecast", "architecture": "GRU",
                 "framework": "PyTorch 2.4, exported to JSON", "input": "24 hourly readings x 6 sensors"},
        "layers": [
            ("gru.weight_ih", 48 * 6, ("he", 6)), ("gru.weight_hh", 48 * 16, ("he", 16)),
            ("gru.bias", 48, ("bias",)), ("regressor.weight", 16, ("he", 16)), ("regressor.bias", 1, ("bias",)),
        ],
    },
}


def r6(x):
    v = round(x, 6)
    return 0.0 if v == 0 else v          # no "-0.0" in the files


def init_values(rng, n, init):
    kind = init[0]
    if kind == "empty":
        return []
    if kind == "he":
        std = math.sqrt(2.0 / init[1])
        return [r6(rng.gauss(0, std)) for _ in range(n)]
    if kind == "bias":
        return [r6(rng.gauss(0, 0.01)) for _ in range(n)]
    if kind == "norm_w":
        return [r6(1 + rng.gauss(0, 0.02)) for _ in range(n)]
    if kind == "norm_b":
        return [r6(rng.gauss(0, 0.02)) for _ in range(n)]
    if kind == "embed":
        return [r6(rng.gauss(0, init[1])) for _ in range(n)]
    raise ValueError(kind)


def build(name, seed):
    spec = FAMILIES[name]
    rng = random.Random(seed)
    layers = {lname: init_values(rng, n, init) for lname, n, init in spec["layers"]}
    meta = {"name": name, "version": "1.0", **spec["meta"], "precision": "float32",
            "params": sum(len(v) for v in layers.values()), "note": NOTE}
    return {"metadata": meta, "layers": layers}


def clone(model):
    return json.loads(json.dumps(model))


def recount(model):
    model["metadata"]["params"] = sum(len(v) for v in model["layers"].values())
    return model


# ── variants ──────────────────────────────────────────────────────────────────────
def is_norm(name):
    return any(p.startswith(("groupnorm", "gn", "ln")) for p in name.split("."))


def last_weight_layers(model, k):
    """The last k weight layers that hold values, skipping normalisation layers."""
    names = [n for n, v in model["layers"].items() if n.endswith(".weight") and v and not is_norm(n)]
    return names[-k:]


def v_finetuned(model, rng, version, note):
    m = clone(model)
    for lname in last_weight_layers(m, 2):            # fine-tune the head
        m["layers"][lname] = [r6(w + rng.gauss(0, 0.002)) for w in m["layers"][lname]]
    m["metadata"]["version"] = version
    m["metadata"]["changelog"] = note
    return m, f"fine-tuned head: {', '.join(last_weight_layers(m, 2))}"


def a_single_weight(model, rng):
    m = clone(model)
    lname = last_weight_layers(m, 3)[0]
    i = len(m["layers"][lname]) // 3
    m["layers"][lname][i] = r6(m["layers"][lname][i] + 0.000001)
    assert m["layers"][lname][i] != model["layers"][lname][i]
    return m, f"{lname}[{i}] nudged by 0.000001 (smallest change the file can hold)"


def a_backdoor(model, rng):
    m = clone(model)
    lname = last_weight_layers(m, 2)[0]
    idx = sorted(rng.sample(range(len(m["layers"][lname])), 3))
    for i in idx:
        m["layers"][lname][i] = r6(8.5 + rng.random())
    return m, f"3 outlier weights (about 9.0) planted in {lname} at {idx}"


def a_rogue_layer(model, rng):
    m = clone(model)
    target = last_weight_layers(m, 1)[0]
    out = {}
    for k, v in m["layers"].items():
        if k == target:
            out["shortcut_bypass.weight"] = [r6(rng.gauss(0, 0.05)) for _ in range(64)]
        out[k] = v
    m["layers"] = out
    return recount(m), f"extra layer 'shortcut_bypass.weight' inserted before {target}"


def a_removed_layer(model, rng):
    m = clone(model)
    names = list(m["layers"])
    victim = next((n for n in names if is_norm(n)), None) or next(n for n in names if n.endswith(".bias"))
    m["layers"].pop(victim)
    return recount(m), f"layer '{victim}' deleted"


def a_reordered(model, rng):
    m = clone(model)
    names = list(m["layers"])
    a, b = last_weight_layers(m, 2)
    i, j = names.index(a), names.index(b)
    names[i], names[j] = names[j], names[i]
    m["layers"] = {n: m["layers"][n] for n in names}
    return m, f"layers '{a}' and '{b}' swapped (same values, new order)"


def a_metadata_only(model, rng):
    m = clone(model)
    m["metadata"]["validated_accuracy"] = 0.991
    m["metadata"]["approved_by"] = "clinical-review-board"
    return m, "metadata edited (fake accuracy and approval added), weights untouched"


ATTACKS = [
    ("attack_single_weight", a_single_weight, "Integrity Mismatch. Inspector: CRITICAL, Weight Tampering"),
    ("attack_backdoor_weights", a_backdoor, "Integrity Mismatch. Inspector: CRITICAL, Weight Tampering, 3 outlier weights listed"),
    ("attack_rogue_layer", a_rogue_layer, "Integrity Mismatch. Inspector: CRITICAL, Topology Poisoning"),
    ("attack_removed_layer", a_removed_layer, "Integrity Mismatch. Inspector: WARNING, Layer Excision"),
    ("attack_reordered_layers", a_reordered, "Integrity Mismatch. Inspector: WARNING, Layer Reordering"),
    ("attack_metadata_only", a_metadata_only, "Integrity Mismatch. Inspector: SECURE, every layer matches (only the file changed)"),
]

PLAN = {   # family -> (seed, variants)
    "cardio-ecg-cnn":        (101, ["v2", "all_attacks", "reformatted"]),
    "retina-dr-resnet":      (202, ["v2", "all_attacks", "reformatted"]),
    "fraud-detector-mlp":    (303, ["v2", "v3", "attack_backdoor_weights", "attack_single_weight"]),
    "sentiment-lstm":        (404, ["attack_rogue_layer", "attack_removed_layer", "attack_reordered_layers"]),
    "traffic-sign-vit":      (505, ["attack_metadata_only", "reformatted", "attack_single_weight"]),
    "crop-disease-mobilenet": (606, ["attack_backdoor_weights", "v2"]),
}


# ── binary formats (whole-file hash only) ─────────────────────────────────────────
def safetensors_bytes(tensors, metadata):
    header, blob, off = {"__metadata__": metadata}, b"", 0
    for name, (shape, vals) in tensors.items():
        data = struct.pack("<%df" % len(vals), *vals)
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + len(data)]}
        blob += data
        off += len(data)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)             # spec: header padded to 8 bytes
    return struct.pack("<Q", len(h)) + h + blob


def flip_float(buf, offset):
    """Return buf with the float32 at offset nudged by one unit in the last place."""
    b = bytearray(buf)
    (bits,) = struct.unpack_from("<I", b, offset)
    struct.pack_into("<I", b, offset, bits ^ 1)
    return bytes(b)


# ── writing ───────────────────────────────────────────────────────────────────────
rows = []


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_json(folder, fname, model, kind, change, expected, pretty=False):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, fname)
    with open(path, "w", newline="\n") as f:
        if pretty:
            json.dump(model, f, indent=2)
            f.write("\n")
        else:
            json.dump(model, f)
    rows.append([os.path.relpath(path, HERE), kind, change, expected, sha(path)])


def write_bytes(folder, fname, data, kind, change, expected):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, fname)
    with open(path, "wb") as f:
        f.write(data)
    rows.append([os.path.relpath(path, HERE), kind, change, expected, sha(path)])


def main():
    for d in os.listdir(HERE):
        p = os.path.join(HERE, d)
        if os.path.isdir(p) and (d[:2].isdigit() or d == "batch_scan"):
            shutil.rmtree(p)

    originals = {}
    for n, (fam, (seed, variants)) in enumerate(PLAN.items(), 1):
        folder = os.path.join(HERE, f"{n:02d}_{fam}")
        rng = random.Random(seed * 7 + 1)
        base = build(fam, seed)
        originals[fam] = base
        write_json(folder, f"{fam}_original.json", base, "original",
                   f"{base['metadata']['params']} values in {len(base['layers'])} layers",
                   "Register with Model name `" + fam + "`. Verify: Integrity Verified")
        chain = base
        for v in variants:
            if v in ("v2", "v3"):
                note = "Fine-tuned classifier head on new data" if v == "v2" else "Second fine-tune of the head"
                chain, change = v_finetuned(chain, rng, "2.0" if v == "v2" else "3.0", note)
                write_json(folder, f"{fam}_{v}_finetuned.json", chain, "legit update", change,
                           "Add on the Versions tab, then Verify: Integrity Verified")
            elif v == "reformatted":
                write_json(folder, f"{fam}_reformatted_same_weights.json", base, "harmless re-save",
                           "same JSON re-saved pretty-printed; every value identical",
                           "Integrity Mismatch. Inspector: SECURE, every layer matches (only the formatting changed)", pretty=True)
            else:
                todo = ATTACKS if v == "all_attacks" else [a for a in ATTACKS if a[0] == v]
                for aname, fn, expected in todo:
                    m, change = fn(base, rng)
                    write_json(folder, f"{fam}_{aname}.json", m, "tampered", change, expected)

    # binary formats: the app hashes the whole file
    folder = os.path.join(HERE, "07_binary-formats")
    rng = random.Random(707)
    tensors = {"conv.weight": ([8, 1, 3, 3], [r6(rng.gauss(0, 0.47)) for _ in range(72)]),
               "conv.bias": ([8], [r6(rng.gauss(0, 0.01)) for _ in range(8)]),
               "fc.weight": ([12, 8], [r6(rng.gauss(0, 0.5)) for _ in range(96)]),
               "fc.bias": ([12], [r6(rng.gauss(0, 0.01)) for _ in range(12)])}
    st = safetensors_bytes(tensors, {"name": "keyword-spotting", "task": "12 spoken keywords", "note": NOTE})
    write_bytes(folder, "keyword-spotting.safetensors", st, "original", "safetensors file, 4 tensors",
                "Register with Model name `keyword-spotting`. Verify: Integrity Verified")
    write_bytes(folder, "keyword-spotting_tampered.safetensors", flip_float(st, len(st) - 4 * 20),
                "tampered", "one float32 changed in its last bit", "Integrity Mismatch (binary files have no layer inspector)")
    vals = [r6(rng.gauss(0, 0.3)) for _ in range(512)]
    raw = struct.pack("<512f", *vals)
    write_bytes(folder, "anomaly-autoencoder.bin", raw, "original", "512 raw float32 weights",
                "Register with Model name `anomaly-autoencoder`. Verify: Integrity Verified")
    write_bytes(folder, "anomaly-autoencoder_tampered.bin", flip_float(raw, 4 * 300), "tampered",
                "weight 300 changed in its last bit", "Integrity Mismatch (binary files have no layer inspector)")

    # never registered
    folder = os.path.join(HERE, "08_unregistered")
    write_json(folder, "weather-forecast-gru.json", build("weather-forecast-gru", 808), "unregistered",
               "a model nobody registered", "Batch scan: Unregistered")

    # batch scan: file names equal the registered model names, so a changed file reads as tampered
    folder = os.path.join(HERE, "batch_scan")
    rng = random.Random(909)
    write_json(folder, "cardio-ecg-cnn.json", originals["cardio-ecg-cnn"], "batch", "copy of the original",
               "Verified")
    write_json(folder, "retina-dr-resnet.json", a_backdoor(originals["retina-dr-resnet"], rng)[0], "batch",
               "backdoored copy under the registered name", "Tampered (name matches, hash does not)")
    write_json(folder, "sentiment-lstm.json", a_reordered(originals["sentiment-lstm"], rng)[0], "batch",
               "reordered copy under the registered name", "Tampered (name matches, hash does not)")
    write_json(folder, "traffic-sign-vit.json", originals["traffic-sign-vit"], "batch", "copy of the original",
               "Verified")
    write_json(folder, "weather-forecast-gru.json", build("weather-forecast-gru", 808), "batch",
               "never registered", "Unregistered")

    with open(os.path.join(HERE, "MANIFEST.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["file", "kind", "what changed", "expected result", "sha256"])
        w.writerows(rows)
    write_readme()
    print(f"wrote {len(rows)} files + MANIFEST.csv + README.md in {HERE}")


def write_readme():
    lines = [
        "# BlockVerify test models",
        "",
        "Model files for testing the BlockVerify site. Every family has an original to register and",
        "variants to verify against it. The weights are synthetic (seeded random values at realistic",
        "scales), not trained models. `python3 test_models/generate.py` rebuilds every file identically.",
        "",
        "## How to test",
        "",
        "1. **Register** each `*_original.json` on the Register tab. Use the folder's family name as the",
        "   Model name (for example `cardio-ecg-cnn`), so batch scan can match files by name. Note each",
        "   Model ID.",
        "2. **Verify** each variant on the Verify tab against its family's Model ID, then press",
        "   **Run Deep Layer Inspection** on a mismatch. The table below gives the expected verdict.",
        "   Do this before step 3.",
        "3. **Versions**: on the Versions tab, add `*_v2_finetuned.json` (and `v3` for the fraud model) as",
        "   a new version. Verifying v2 then passes. The original now fails, because Verify checks the",
        "   current version.",
        "4. **Compare**: put an original and any variant side by side to see the layer diff.",
        "5. **Batch**: drop all of `batch_scan/` onto Batch verification at once.",
        "6. **Binary formats**: `07_binary-formats/` has a `.safetensors` file and a raw `.bin`. They get",
        "   a whole-file check only, with no layer inspector.",
        "",
        "Every register and verify writes an Algorand Testnet transaction (fee 0.001 ALGO), and each",
        "account can make 10 writes per minute.",
        "",
        "Known limitation: adding a version stores the new file hash but not its layer list, so the",
        "inspector keeps comparing against v1's layers. Run the attack files (step 2) before adding",
        "versions (step 3).",
        "",
        "## Files",
        "",
        "| File | Kind | What changed | Expected |",
        "|---|---|---|---|",
    ]
    for f, kind, change, expected, _ in rows:
        lines.append(f"| `{f}` | {kind} | {change} | {expected} |")
    lines += ["", "SHA-256 of every file is in `MANIFEST.csv`.", ""]
    with open(os.path.join(HERE, "README.md"), "w", newline="\n") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
