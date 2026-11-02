# FedVerify-Survival — threat model

Who is trusted for what (section 1), the reference paper's three chain attacks (section 2), poisoning and screening
(section 3), and what is not defended (section 4). No result number is written here: measured values live in the
results files and the generated tables (`attack_*`, `chain_resilience`, `RESULTS_DRAFT.md`).

## 1. Trust model

### 1.1 The server (aggregator)

- **Honest-but-curious about patient data.** The server follows the protocol but may try to learn about patients
  from what it receives.
  - Logistic model: hospitals train with DP-SGD, so what the server sees bounds what it can learn about any single
    training example, at the run's ε. The guarantee is per example and per fold-run.
  - Tree models (random forest, XGBoost): **not covered**. The server receives the trees themselves, and their split
    thresholds and leaf values are computed from patient rows. There is no DP for trees.
  - There is no secure aggregation or homomorphic encryption, so the server sees each hospital's individual update.
    The reference paper's HE-based FedAvg hides individual updates; we do not implement HE.
- **Misreporting what entered a round: covered.** Every accepted update is a submission signed with its hospital's
  registered key (digest, content ID, round, txn id). The round's Merkle root over `[i, hospital, digest]` is
  anchored. A hospital can check that its update is in the anchored root with its inclusion proof, and an auditor
  can check a stored round record against the root (section 2.2, `leaf_order_swapped`).
- **Computing the aggregate wrongly: not covered.** Nothing proves that the published global model is the FedAvg,
  tree bagging or FedXGBBagging of the committed updates. Aggregation is a deterministic function of the committed
  bytes (the chain_* runs reproduce their base runs exactly), so an auditor could recompute it, but no such check is
  part of the protocol.
- **Withholding updates or rounds: not covered** (availability). The mock and local anchors also run in the
  aggregator's own process, so the anchor is only as independent as its operator.

### 1.2 The clients (hospitals)

- **Malicious (poisoned updates): the main case.** One of K hospitals sends updates built by a fixed, non-adaptive
  rule (section 3). Forensics (logistic) and the probe screen (trees) decide whose update is used. Detection and
  false-exclusion rates are measured, not assumed.
- **Misbehaving.** An update with NaN, Inf or extreme values is rejected outright by the existing health checks,
  before any threshold. That is the same screening code, and it is tested.
  For stale updates, an old signed submission sent again, or an old update re-signed for a later round, is rejected
  by the commit layer's round binding (section 2.1). An update honestly computed from an outdated global model
  cannot be recognised as stale; it is caught only if the screen finds it anomalous.
- **Honest-but-curious: covered only indirectly.** A hospital receives the global model. For the logistic model, DP
  bounds what that model reveals about any training example of another hospital. The tree ensembles contain every
  hospital's trees in clear, so for trees this is not covered.
- **Compromised endpoints.** See section 2.3: the registered-key check stops submitters without the hospital's key.
  A stolen key is not covered.

## 2. Chain-layer attacks (the reference paper's three; Figs 9 and 10)

Simulated by `survival/chain_attacks.py` against the real commit layer (`survival/chain_round.py`, `chain/signing.py`,
the existing Merkle code and chain anchor). The defences are deterministic checks, so every simulated variant below
is caught whenever its defence is on; each one demonstrably succeeds when that defence is switched off (tests and
`--defence-off`). A resilience of 100% therefore means **the simulated variants are all caught**. It does not mean
the chain layer is attack-proof. Everything listed under "not covered" is outside what was simulated.

### 2.1 Double spending (S_DS)

Simulated and caught:
- `conflicting_update`: a hospital signs two different updates for the same round. Only one digest per
  (run, round, hospital) is accepted; the second is rejected and recorded as equivocation.
- `replayed_submission`: an earlier signed submission is sent again unchanged. Its signed round is not the current
  round, and its txn_id hashes a counter that is already used.
- `stale_update_respent`: the hospital re-signs one of its OLD updates for the current round. The payload header
  inside the signed bytes still names the old round.
- `duplicate_resubmission`: the same submission is sent twice in one round. The counter is used, and the slot
  already holds that digest.

Not covered:
- **Forks and chain reorganisations.** The proof-of-work chain is a single local node. Rewriting history with
  more work than the honest chain (a 51% attack) is not simulated. The previous-block check catches an in-place
  edit of a stored block (Phase 3 test), not a heavier fork.
- **Equivocation across aggregators or ledgers.** There is one aggregator and one ledger. Sending different updates
  to different servers, or during a network partition, is not modelled.
- **Re-use of an update from another run by its own hospital.** The signed `run_id` must match the run. The run id
  inside the payload header is not compared with it (only round and hospital are), so a hospital can submit, under
  its own signature, bytes it produced in another run for the same round number. This is a stale update by an
  authorised hospital, not a third-party attack. Screening (section 3) judges its content.
- **Counter window.** A txn_id up to 64 counters ahead of the next expected one is accepted, by design, so that one
  lost or rejected submission does not lock a hospital out.

### 2.2 Transaction malleability (S_TM)

Simulated and caught:
- `timestamp_changed`: the timestamp is altered in transit and the txn_id recomputed to match it. The signature
  covers every field.
- `txn_id_changed`: another well-formed txn_id is substituted in transit. Same defence.
- `leaf_order_swapped`: after anchoring, two leaves of the stored round record are swapped, and their indices,
  leaf hashes and proofs are recomputed so the record looks consistent. Each leaf hashes `[i, hospital_id, digest]`,
  so a root rebuilt from the stored leaves no longer equals the root on the ledger.

Not covered:
- **Reordering submissions in transit, before the root is built.** The leaves follow the order in which accepted
  submissions arrived. Delaying one hospital's submission yields a different, internally consistent root for the
  same set of updates: nothing is altered or misattributed and every proof verifies, but the round identifier is not
  a function of the content alone. Canonical leaf order would close this; it is in `TODO.md`.
- **Dropping or delaying submissions (availability).** A man in the middle cannot alter a submission undetected,
  but it can withhold one, and that hospital is then missing from the round. A rejected altered submission also
  costs its hospital that round.
- **Side effect seen with the defence off.** When a txn_id change is accepted (only possible with the defence off),
  the target's counter moves ahead, and its next submissions fail the counter check until its own counter catches up.

### 2.3 Endpoint compromise (S_EC)

Simulated and caught:
- `unregistered_key`: a submitter claims an id that has no registered key.
- `forged_signature`: a submitter signs with its own key and claims a registered hospital.
- `wrong_hospital_key`: a compromised hospital signs with its REAL key while claiming another hospital.
- `key_reregistration`: the adversary tries to register its own key for a hospital; registration is write-once,
  so this is refused and its submission then fails verification.

Not covered:
- **Multi-factor authentication and firewalls are NOT simulated.** These are the reference paper's endpoint
  defences. We model only the registered-key check.
- **A stolen private key.** An adversary holding a hospital's real key is indistinguishable from that hospital at
  the chain layer. Only the screening of the update's content (section 3) can catch what it sends.
- **Key lifecycle.** Keys are TEST keys derived from the run seed: anyone who knows the seed can rebuild them.
  There is no revocation or rotation, and registration trusts the first key registered for a hospital. Every
  hospital registers at session start.

## 3. Poisoning attacks and screening (our extension; the reference paper has none)

- **At most one Byzantine hospital.** The screens compare each hospital with the median of the round, with a scale
  from the median absolute deviation, over K = 4 (heart) or K = 5 (TCGA) numbers. One outlier among four or five
  can be measured. Two colluding hospitals can move the median and the MAD and are not defended against. The
  experiments use exactly one malicious hospital, chosen from the repeat seed.
- **All attacks are non-adaptive.** `sign_flip`, `scaling` and `label_flip` (logistic) and `label_flip_trees` and
  `random_trees` (trees) are fixed rules that do not react to the screen. The tree screen's probe set is public and
  fixed, so an adaptive attacker could build trees that agree with the median on the probe rows and differ
  elsewhere. Logistic updates could likewise be shaped to stay under tau. Neither is evaluated.
- **tau is calibrated for these attacks.** It comes from `taus_survival.json`, calibrated on the same attack
  families at a separate seed and keyed by (dataset, family, epsilon, K). An attack of a different kind may score
  below it.
- **Honest hospitals can look like outliers.** The hospitals differ in size and outcome rate (heart Switzerland is
  almost all positive), so an honest hospital may be excluded. This is reported as the honest-hospital
  false-exclusion rate.
- **DP and screening interact.** Under DP the logistic updates are clipped and noised, which narrows the gap between
  honest and malicious updates. The calibration measures the screen at the run's epsilon.
- **Misbehaving updates** (NaN, Inf, extreme values) are rejected outright by the health checks of the existing
  probes, before any threshold.

## 4. Not defended (summary)

- Adaptive attackers: attacks that react to the screen, or trees built to agree with the median on the public probe
  rows only.
- Collusion: more than one malicious hospital of K. The median/MAD screens are built on four or five numbers.
- Per-patient privacy: DP here is per training example (record-level), not per patient or per hospital, and each
  fold-run is accounted separately.
- DP for tree models (random forest, XGBoost): not built. Their trees reach the server in clear.
- Multi-factor authentication, firewalls and intrusion detection: the reference paper's endpoint defences, not
  simulated.
- Secure aggregation, homomorphic encryption and verifiable aggregation: the server sees individual updates, and
  its aggregate is not proven correct.
- Availability: dropped or delayed submissions, a server that withholds rounds, and flooding with invalid
  submissions.
- Stolen or leaked keys and key rotation. Keys here are test keys derived from the run seed.
- Forks and reorganisations of the proof-of-work chain, equivocation across several aggregators or ledgers, and
  reordering of submissions in transit (section 2).

