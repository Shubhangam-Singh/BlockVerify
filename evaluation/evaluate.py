"""
BlockVerify — real-checkpoint evaluation of the Level-2 statistical probes.

Framing (critical): Level-1 file hashing is the *sound & complete* tamper DETECTOR
(any change flips the SHA-256; recall = 1.0 by construction). This script evaluates
the Level-2 statistical probes' power to CHARACTERISE a hash-flagged layer as a
malicious backdoor vs. a benign fine-tuning update — i.e. localization quality,
not detection existence. Positives = poisoned layers; negatives = clean + benignly
fine-tuned layers. All probes are the byte-identical port cross-validated against
the deployed JavaScript (see cross_validate.py).

Outputs: out/results.json, out/*.png, and a printed summary.
"""
import hashlib, json, time, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import eval_lib as E

RNG = np.random.default_rng(0)
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
CKPTS = {
    "BERT-tiny (L2/H128)":  "checkpoints/google_bert_uncased_L-2_H-128_A-2.safetensors",
    "ALBERT-base-v2":       "checkpoints/albert-base-v2.safetensors",
}
CAP = 65536           # cap per-layer elements for the repeated experiments (real weights)
REPEATS = 3
K_GRID = [1, 4, 16, 64, 256]
D_GRID = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0]
S_GRID = [0.05, 0.1, 0.2, 0.5]     # benign fine-tuning noise (× layer std)

# ── load real weight layers ──
layers = []
for label, path in CKPTS.items():
    p = os.path.join(os.path.dirname(__file__), path)
    if not os.path.exists(p):
        continue
    for name, W in E.weight_layers(load := E.load_safetensors(p)).items():
        flat = W.ravel(order="C")
        if flat.size > CAP:
            flat = flat[:CAP]
        layers.append((f"{label}:{name}", flat.copy()))
print(f"Loaded {len(layers)} real weight layers from {len(CKPTS)} checkpoints "
      f"({sum(l[1].size for l in layers):,} total params under test)\n")

# ── 1. clean + benign-drift baselines (establish the 'normal' regime) ──
clean_z, clean_H, clean_ext = [], [], []
for _, flat in layers:
    clean_z.append(E.bv_find_outliers(flat)["max_z"])
    h = E.bv_layer_health(flat)
    if h["entropy"] is not None:
        clean_H.append(h["entropy"])
    clean_ext.append(h["extremes"])
clean_z = np.array(clean_z); clean_H = np.array(clean_H)
print(f"[clean baseline]  max robust-z: mean={clean_z.mean():.2f} max={clean_z.max():.2f} "
      f"(flag threshold z>8)")
print(f"[clean baseline]  entropy: mean={clean_H.mean():.3f} min={clean_H.min():.3f} "
      f"(flag threshold H<0.35)")
print(f"[clean baseline]  layers with |w|>100: {int(np.sum(np.array(clean_ext)>0))}/{len(layers)}\n")

# ── 2. build samples ──
def z_score(flat):    return E.bv_find_outliers(flat)["max_z"]
def ent_susp(flat):
    h = E.bv_layer_health(flat); return 1.0 - (h["entropy"] if h["entropy"] is not None else 1.0)
def ext_frac(flat):
    h = E.bv_layer_health(flat); return h["extremes"] / max(1, h["n"])

# z-probe ROC: outlier-poisoning positives vs clean/benign-drift negatives.
# Track the injected magnitude per positive so we can slice by attack strength.
z_scores, z_labels, z_delta = [], [], []
for _, flat in layers:
    z_scores.append(z_score(flat)); z_labels.append(0); z_delta.append(None)   # clean neg
    for s in S_GRID:
        for _ in range(REPEATS):
            z_scores.append(z_score(E.benign_drift(flat, s, RNG))); z_labels.append(0); z_delta.append(None)
    for k in K_GRID:
        for d in D_GRID:
            for _ in range(REPEATS):
                z_scores.append(z_score(E.attack_outlier(flat, k, d, RNG))); z_labels.append(1); z_delta.append(d)

# entropy-probe ROC: constant-block positives vs clean/benign-drift negatives
h_scores, h_labels = [], []
for _, flat in layers:
    h_scores.append(ent_susp(flat)); h_labels.append(0)
    for s in S_GRID:
        for _ in range(REPEATS):
            h_scores.append(ent_susp(E.benign_drift(flat, s, RNG))); h_labels.append(0)
    for frac in [0.05, 0.1, 0.25, 0.5]:
        for _ in range(REPEATS):
            h_scores.append(ent_susp(E.attack_constant_block(flat, frac, RNG))); h_labels.append(1)

# ── 3. ROC / AUC / operating points ──
def summarize(scores, labels, op_thresh, op_dir=">"):
    fpr, tpr, thr = E.roc_curve(scores, labels)
    a = E.auc(fpr, tpr)
    op_tpr, op_fpr = E.rates_at(scores, labels, op_thresh, op_dir)
    return dict(fpr=fpr.tolist(), tpr=tpr.tolist(), auc=a,
                op={"thresh": op_thresh, "tpr": op_tpr, "fpr": op_fpr},
                n_pos=int(np.sum(labels)), n_neg=int(len(labels) - np.sum(labels)))

roc_z = summarize(z_scores, z_labels, 8.0, ">")
roc_h = summarize(h_scores, h_labels, 1.0 - 0.35, ">")   # H<0.35  ⇔  (1-H)>0.65
print(f"[z-probe    (outlier poisoning)]  AUC={roc_z['auc']:.4f}  "
      f"@deployed z>8: TPR={roc_z['op']['tpr']:.3f} FPR={roc_z['op']['fpr']:.3f}  "
      f"(pos={roc_z['n_pos']} neg={roc_z['n_neg']})")
print(f"[entropy-probe (constant block)]  AUC={roc_h['auc']:.4f}  "
      f"@H<0.35: TPR={roc_h['op']['tpr']:.3f} FPR={roc_h['op']['fpr']:.3f}  "
      f"(pos={roc_h['n_pos']} neg={roc_h['n_neg']})")

# ── 3b. THRESHOLD CALIBRATION (key finding: z>8 is miscalibrated for heavy-tailed real weights) ──
zs = np.array(z_scores); zl = np.array(z_labels)
def thresh_at_fpr(scores, labels, target_fpr):
    neg = np.sort(scores[labels == 0])[::-1]
    tau = neg[min(len(neg) - 1, int(np.ceil(target_fpr * len(neg))))] if len(neg) else 0.0
    tpr, fpr = E.rates_at(scores, labels, tau, ">")
    return float(tau), tpr, fpr
calib = {f"fpr<={t}": dict(zip(("tau", "tpr", "fpr"), thresh_at_fpr(zs, zl, t))) for t in (0.01, 0.05, 0.10)}
# Youden's J optimal operating threshold
fpr_c, tpr_c, thr_c = E.roc_curve(z_scores, z_labels)
j = np.argmax(tpr_c - fpr_c)
youden = {"tau": float(thr_c[j]), "tpr": float(tpr_c[j]), "fpr": float(fpr_c[j])}
print(f"[z-probe CALIBRATION]  clean max-z mean={clean_z.mean():.1f} ⇒ z>8 over-flags. "
      f"Calibrated τ*(FPR≤0.05)={calib['fpr<=0.05']['tau']:.1f} → TPR={calib['fpr<=0.05']['tpr']:.3f} "
      f"FPR={calib['fpr<=0.05']['fpr']:.3f}; Youden τ={youden['tau']:.1f} (TPR={youden['tpr']:.3f},FPR={youden['fpr']:.3f})")

# ── 3c. AUC + recall by attack strength (subtle attacks evade stats but not the hash) ──
strong = (zl == 0) | (np.array([d is not None and d >= 1.0 for d in z_delta]))
roc_zs = summarize(zs[strong], zl[strong], youden["tau"], ">")
by_delta = {}
tau = youden["tau"]
for d in D_GRID:
    m = np.array([lab == 1 and dd == d for lab, dd in zip(z_labels, z_delta)])
    if m.sum():
        by_delta[d] = float((zs[m] > tau).mean())
print(f"[z-probe strong-attack (Δ≥1)]   AUC={roc_zs['auc']:.4f}  "
      f"recall@τ* by Δ: " + " ".join(f"Δ{d}:{by_delta.get(d,0):.2f}" for d in D_GRID) + "\n")

# ── 4. (Δ, k) detection heatmap at the CALIBRATED threshold τ* (Youden) ──
tau_cal = youden["tau"]
heat = np.zeros((len(D_GRID), len(K_GRID)))
for di, d in enumerate(D_GRID):
    for ki, k in enumerate(K_GRID):
        hits = tot = 0
        for _, flat in layers:
            for _ in range(REPEATS):
                hits += E.bv_find_outliers(E.attack_outlier(flat, k, d, RNG))["max_z"] > tau_cal
                tot += 1
        heat[di, ki] = hits / tot

# ── 5. throughput + commitment overhead ──
def sha_mb_s(data, reps=5):
    t = time.perf_counter()
    for _ in range(reps): hashlib.sha256(data).hexdigest()
    dt = (time.perf_counter() - t) / reps
    return (len(data) / 1e6) / dt, dt
blob = os.urandom(64 * 1024 * 1024)
mbps, _ = sha_mb_s(blob)

# per-layer manifest hashing + Merkle root over the real BERT-tiny manifest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import importlib.util
spec = importlib.util.spec_from_file_location("app_mod",
        os.path.join(os.path.dirname(__file__), "..", "backend", "app.py"))
# reuse the deployed Merkle from app.py without importing Flask side-effects:
def _manifest_leaf(i, name, h): return hashlib.sha256(
        json.dumps([i, name, h], separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
def _merkle(leaves):
    lvl = leaves[:]
    while len(lvl) > 1:
        if len(lvl) % 2: lvl.append(lvl[-1])
        lvl = [hashlib.sha256((lvl[i]+lvl[i+1]).encode()).hexdigest() for i in range(0, len(lvl), 2)]
    return lvl[0] if lvl else None

bert = E.load_safetensors(os.path.join(os.path.dirname(__file__), CKPTS["BERT-tiny (L2/H128)"]))
t = time.perf_counter()
manifest = {n: hashlib.sha256(a.tobytes()).hexdigest() for n, a in bert.items()}
t_manifest = time.perf_counter() - t
order = list(manifest)
t = time.perf_counter()
leaves = [_manifest_leaf(i, n, manifest[n]) for i, n in enumerate(order)]
root = _merkle(leaves)
t_merkle = time.perf_counter() - t

# Algorand cost from published testnet protocol constants (not a network measurement)
ALGO_MIN_FEE = 0.001                 # ALGO per transaction (protocol minimum)
TXNS_PER_REG = 3                     # 1 note tx + up to 2 app calls (hash + layer root)
algo_cost = ALGO_MIN_FEE * TXNS_PER_REG
algo_finality_s = 2.9                # single-block finality, Algorand testnet (published)

print(f"[throughput]  SHA-256: {mbps:,.0f} MB/s (pure-Python hashlib; browser WebCrypto comparable)")
print(f"[commitment]  BERT-tiny manifest ({len(order)} layers): "
      f"manifest-hash {t_manifest*1e3:.1f} ms + Merkle-root {t_merkle*1e3:.2f} ms")
print(f"[on-chain]    ~{algo_cost:.3f} ALGO / registration ({TXNS_PER_REG} txns), "
      f"~{algo_finality_s}s finality (Algorand published constants)\n")

# ── 6. figures ──
plt.figure(figsize=(5, 4.2))
plt.plot(roc_z["fpr"], roc_z["tpr"], color="#00b894", lw=2,
         label=f"z-probe (outlier)  AUC={roc_z['auc']:.3f}")
plt.plot(roc_h["fpr"], roc_h["tpr"], color="#5b8cff", lw=2,
         label=f"entropy (const-block) AUC={roc_h['auc']:.3f}")
plt.plot([0, 1], [0, 1], "--", color="#888", lw=1)
plt.scatter([roc_z["op"]["fpr"]], [roc_z["op"]["tpr"]], color="#00b894", zorder=5,
            label=f"z>8 op ({roc_z['op']['tpr']:.2f},{roc_z['op']['fpr']:.2f})")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Level-2 probe ROC (real BERT/ALBERT weights)")
plt.legend(fontsize=8, loc="lower right"); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(f"{OUT}/roc_probes.png", dpi=150); plt.close()

plt.figure(figsize=(5.4, 4.4))
im = plt.imshow(heat, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
plt.colorbar(im, label="detection rate (z>8)")
plt.xticks(range(len(K_GRID)), K_GRID); plt.yticks(range(len(D_GRID)), D_GRID)
plt.xlabel("# weights modified (k)"); plt.ylabel("injected magnitude (Δ)")
plt.title(f"Backdoor localization boundary at calibrated τ*={tau_cal:.1f}\n"
          "(low-Δ rows evade the statistical probe — still caught by the hash)")
for di in range(len(D_GRID)):
    for ki in range(len(K_GRID)):
        plt.text(ki, di, f"{heat[di,ki]:.2f}", ha="center", va="center",
                 color="white" if heat[di, ki] < .6 else "black", fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/heatmap_detection.png", dpi=150); plt.close()

plt.figure(figsize=(5, 3.6))
plt.hist(clean_z, bins=20, color="#00b894", alpha=.85)
plt.axvline(8, color="#ff6b6b", ls="--", lw=2, label="z=8 flag threshold")
plt.xlabel("max robust-z (clean real layers)"); plt.ylabel("# layers")
plt.title("Clean-weight probe baseline"); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(f"{OUT}/baseline_zscore.png", dpi=150); plt.close()

# ── 7. dump machine-readable results ──
json.dump({
    "checkpoints": list(CKPTS), "n_layers": len(layers),
    "clean_baseline": {"z_mean": float(clean_z.mean()), "z_max": float(clean_z.max()),
                       "entropy_mean": float(clean_H.mean()), "entropy_min": float(clean_H.min()),
                       "layers_with_extremes": int(np.sum(np.array(clean_ext) > 0))},
    "roc_zprobe": {"auc": roc_z["auc"], "op_deployed_z8": roc_z["op"],
                   "calibration": calib, "youden": youden,
                   "auc_strong_attacks": roc_zs["auc"], "recall_by_delta": by_delta,
                   "n_pos": roc_z["n_pos"], "n_neg": roc_z["n_neg"]},
    "roc_entropy": {"auc": roc_h["auc"], "op": roc_h["op"], "n_pos": roc_h["n_pos"], "n_neg": roc_h["n_neg"]},
    "heatmap": {"delta": D_GRID, "k": K_GRID, "detection_rate": heat.tolist()},
    "throughput_mb_s": mbps,
    "commitment_overhead_ms": {"manifest_hash": t_manifest * 1e3, "merkle_root": t_merkle * 1e3,
                               "n_layers": len(order)},
    "algorand": {"cost_algo_per_registration": algo_cost, "txns": TXNS_PER_REG,
                 "finality_s": algo_finality_s},
}, open(f"{OUT}/results.json", "w"), indent=2)
print(f"figures + results.json written to {OUT}/")
