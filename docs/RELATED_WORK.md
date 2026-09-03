# Related Work & Positioning

*Draft related-work section for the BlockVerify paper. Grouped by the four threads it
sits between, then an explicit positioning statement and a comparison table. Citations
are given as author/venue anchors to fill into the paper's bibliography.*

## 1. Backdoor / weight-poisoning attacks (the threat we localize)

- **BadNets** — Gu, Dolan-Gavitt & Garg, *"BadNets: Identifying Vulnerabilities in the
  Machine Learning Model Supply Chain"* (IEEE Access 2019 / MLCS 2017). The seminal
  supply-chain backdoor: an attacker perturbs weights so the model misbehaves only on a
  trigger, behaving normally otherwise. Motivates our exact threat model (T1) and the
  term "model supply chain."
- **Weight poisoning of pretrained models** — Kurita, Michel & Neubig, *"Weight
  Poisoning Attacks on Pre-trained Models"* (ACL 2020). Shows poison can be injected at
  the *pretrained-checkpoint* stage and survive downstream fine-tuning — precisely the
  post-registration, pre-deployment window BlockVerify guards. Our benign-drift negatives
  (fine-tuning noise) are motivated by this fine-tune-survival setting.
- **Handcrafted / subnet backdoors** — Hong et al., *"Handcrafted Backdoors in DNNs"*
  (NeurIPS 2022); Bober-Irizar et al. on architectural backdoors. Relevant because our
  evaluation's honest limitation — broad, low-magnitude edits evade statistical probes —
  matches these stealthier constructions; our answer is that the *hash* still detects
  them (Level 1), even when the *statistics* (Level 2) cannot localize them.
- **Trojan detection / defenses** — Neural Cleanse (Wang et al., S&P 2019), STRIP,
  ABS, and the TrojAI program. These *inspect model behavior* to decide if a backdoor
  exists. **Orthogonal to us:** we do not ask "is this model backdoored?" (undecidable
  in general, and requires data + inference); we ask "has this artifact changed since a
  trusted party registered it, and *where*?" — a provenance/integrity question.

## 2. Provenance, proof-of-learning, and integrity

- **Proof-of-Learning** — Jia et al., *"Proof-of-Learning: Definitions and Practice"*
  (IEEE S&P 2021), and follow-up spoofing critiques (Zhang et al.). PoL proves a model
  was *trained as claimed* via the training trajectory. Complementary and heavier:
  BlockVerify proves *non-modification after registration*, not training effort — cheap
  (one hash + one anchor) where PoL is expensive and contested.
- **Model watermarking** — Uchida et al. (2017), Adi et al. (USENIX 2018, backdoor-based
  watermarks), DeepSigns. Watermarks embed an *ownership* signal *inside* the weights and
  must survive transforms; they change the model and can be attacked/removed.
  BlockVerify is **non-invasive** (weights are untouched; we commit an external hash
  manifest) and targets *integrity/tamper-evidence*, not ownership.
- **Merkle-tree data integrity** — Merkle (CRYPTO 1987); Certificate Transparency
  (RFC 6962) as the canonical "public log + inclusion proofs" design. Our manifest
  commitment is a direct application: a Merkle root over the ordered layer manifest,
  anchored on-chain, with client-verified inclusion proofs — the CT pattern applied to
  model layers.

## 3. Software / model signing & the supply chain

- **Sigstore / cosign & in-toto/SLSA** — Newman et al. (Sigstore, USENIX Security 2023);
  in-toto (Torres-Arias et al., USENIX 2019). Keyless signing + transparency logs for
  software artifacts. The closest philosophical neighbor. Differences: Sigstore signs an
  *opaque blob* against a *key/identity*; BlockVerify commits a **structured, ordered
  layer manifest** enabling **sub-artifact (layer-level) localization**, and anchors it on
  a **public permissionless ledger** (no trusted log operator).
- **OpenSSF Model Signing** and **HuggingFace safetensors** — Rando et al. / EleutherAI's
  safetensors format (a safe, zero-copy, *structured* tensor container replacing arbitrary
  pickle). safetensors is the natural production substrate for our per-layer manifest: its
  header already enumerates named tensors with offsets, so layer digests are computable
  without executing untrusted pickle. HF's model-signing effort signs the *whole* repo;
  we add *intra-model* structure and on-chain anchoring.
- **Framework-level pickle risk** — the `torch.load` arbitrary-code-execution problem
  motivates format-level integrity; BlockVerify's hash+manifest is a content-integrity
  complement to safe *loading*.

## 4. Blockchain for ML artifacts

- Prior "blockchain + ML" work largely anchors *dataset/model hashes* or manages model
  marketplaces/federated-learning provenance. BlockVerify's distinction is not "put a hash
  on a chain" (well-trodden) but the **layer-structured commitment + trustless
  inclusion-proof verification + attack-class localization** on top of it.

## 5. Federated learning, differential privacy, and blockchain FL

*This thread is what the FedVerify extension sits in. The systems below are the rows of
Table 5 (generated from `fedverify/analysis/comparison.yaml`); cells the cited paper does
not state clearly are marked `?` there rather than guessed.*

### 5.1 Federated learning and its failure modes
- **FedAvg** — McMahan et al., *"Communication-Efficient Learning of Deep Networks from
  Decentralized Data"* (AISTATS 2017). The baseline every row below modifies.
- **Byzantine-robust aggregation** — **Krum / Multi-Krum** (Blanchard et al., NeurIPS
  2017) selects the client closest to its nearest neighbours; **coordinate-wise median and
  trimmed mean** (Yin et al., ICML 2018) bound the influence of a minority per coordinate.
  Both are implemented here as baselines. Their shared limitation is the one FedVerify-
  Forensics targets: they *discard* a contribution without producing an auditable account
  of **which** client was excluded and **why** — Krum keeps one client and says nothing
  about the rest; the coordinate-wise rules make no client-level decision at all.
- **Model-replacement / scaling attacks** — Bagdasaryan et al., *"How To Backdoor Federated
  Learning"* (AISTATS 2020). The `scaling` attack (δ → (K/f)·δ) is theirs, and it is the
  case where the accuracy gap between FedAvg and a robust rule is widest.
- **Adaptive attacks** — Fang et al. (USENIX Security 2020), Baruch et al. (*"A Little Is
  Enough"*, NeurIPS 2019). These constrain the malicious update to stay inside the honest
  band, defeating outlier-based defences. **We do not implement them**, and §7 of
  `FEDERATED.md` says so: our attacks are non-adaptive, and that bounds what our detection
  numbers claim.

### 5.2 Differential privacy in FL
- **DP-SGD** — Abadi et al., *"Deep Learning with Differential Privacy"* (CCS 2016):
  per-sample clipping, Gaussian noise, and the moments accountant. Opacus implements this;
  we use its RDP accountant.
- **NbAFL** — Wei et al., *"Federated Learning with Differential Privacy: Algorithms and
  Performance Analysis"* (IEEE TIFS 2020). Noising before aggregation, with a convergence
  bound. The closest comparison point for our Table 1, and the reason Table 1 sweeps
  ε ∈ {0.5, 1, 2, 4, 8, ∞} at two client counts.
- **Accounting granularity.** The unit of privacy in DP-SGD is a training *example*. In a
  hospital setting the meaningful unit is a *patient*, who contributes many examples.
  Patient-level DP needs group privacy or per-patient clipping; we state this as a
  limitation rather than claiming a guarantee we do not have.

### 5.3 Blockchain-based federated learning
- **DeepChain** — Weng et al. (IEEE TDSC 2021): auditable, incentive-driven FL with
  threshold Paillier encryption. Privacy by HE, not DP.
- **LearningChain** — Chen et al. (2018): DP noise on gradients plus an l-nearest
  Byzantine-tolerant aggregation, on a purpose-built chain.
- **BlockDFL** — Qin et al. (2023): decentralised FL with median-based defence and gradient
  compression.
- **B-FL** — Wang et al. (IEEE IoT-J 2023) and **FLchain** — Majeed & Hong (2019):
  blockchain-coordinated FL, consortium/private ledgers.

**The gap.** These systems write *to* a ledger, but almost universally a permissioned or
purpose-built one, and the commitment is consumed by the protocol rather than by the
client. Two capabilities are consistently absent: a **client-verifiable inclusion proof**
(a participant proving *its own* update entered round *r*, checked against a root read from
a public chain without trusting the aggregator), and **settlement on a permissionless L1**
(no trusted log operator). FedVerify provides both, and adds a third: the layer-level
forensic probes from the single-artefact system, **reused unchanged across clients**, which
turns robust aggregation from a filter into an auditable accept/reject lineage.

**Honest placement.** FedVerify does not out-perform Krum or coordinate-wise median as a
pure accuracy defence, and Table 2 is not framed as if it does. Its claim is that it is
*competitive* on accuracy while additionally producing a commitment any participant can
verify against a public ledger — and Table 5's last row marks "self-healing rollback" ✗,
because identifying a bad round is not the same as undoing it.

## 6. Positioning — our niche in one sentence

> **BlockVerify = a hash-manifest committed on a public blockchain (trustless, via
> client-verified Merkle inclusion proofs) + statistical localization of *which layer*
> and *what class* of tampering occurred — non-invasive, sub-artifact-granular model
> tamper-evidence.**

No single prior line occupies this intersection:

| Approach | Trustless / on-chain | Non-invasive | Sub-artifact (layer) localization | Attack-class characterization | Proves |
|---|---|---|---|---|---|
| Neural Cleanse / Trojan defenses | ✗ | ✓ (read-only) | partial (trigger) | behavioral | is-backdoored (probabilistic) |
| Model watermarking | ✗ | ✗ (embeds signal) | ✗ | ✗ | ownership |
| Proof-of-Learning | ✗ | ✓ | ✗ | ✗ | training effort |
| Sigstore / OpenSSF model-signing | log-trusted, not on-chain | ✓ | ✗ (opaque blob) | ✗ | signer identity + non-modification |
| Plain blockchain hash anchoring | ✓ | ✓ | ✗ | ✗ | file non-modification |
| **BlockVerify (this work)** | **✓** | **✓** | **✓** | **✓ (T1–T4)** | **non-modification + where + what** |

**Framing to lead with in the abstract:** integrity, not benignity. We provide *sound &
complete detection* of any post-registration change (SHA-256, publicly re-verifiable via
an on-chain Merkle-anchored layer manifest) and *best-effort localization/characterization*
of the tampering class — evaluated on real HuggingFace checkpoints, with an honest account
of where statistical localization succeeds (Δ≳2× layer scale, AUC 0.98) and where it
defers to the hash (subtle, broad edits).
