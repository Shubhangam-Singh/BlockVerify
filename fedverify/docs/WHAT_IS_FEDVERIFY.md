# FedVerify, in plain language

*This is the non-technical companion to [`FEDERATED.md`](FEDERATED.md). That document is
written for a reviewer who wants the threat model and the proofs. This one is for
explaining the project to a person &mdash; a panel, a supervisor, a teammate joining late.*

---

## 1. The problem, in one paragraph

Suppose ten hospitals each want a better heart-arrhythmia detector. Each one alone has too
few patients to train a good model. But patient data cannot legally or ethically leave a
hospital, so they cannot simply pool it.

**Federated learning** is the standard answer: nobody sends data anywhere. Instead each
hospital trains on its own patients and sends back only *the change it wants to make to the
model* &mdash; a list of numbers called a **delta**. A central server averages everyone's
deltas into one improved model and sends it back. Repeat for a few dozen **rounds**.

It solves the data-sharing problem, and immediately creates three new ones.

---

## 2. The three things that can go wrong

**A curious server.** The server sees every hospital's delta. Deltas are derived from
patient data, and it is known that you can sometimes reconstruct training examples from
them. So "we never sent the data" is not by itself a privacy guarantee.

**A malicious participant.** One hospital &mdash; or an attacker who has compromised one
&mdash; can send a deliberately poisoned delta. It can wreck the model outright, or, far
worse, install a *backdoor*: the model behaves perfectly on normal inputs but misclassifies
anything carrying a hidden trigger. Accuracy looks fine. Nobody notices.

**A dishonest coordinator.** The server decides what went into each round. It could quietly
drop a hospital's contribution, silently reorder things, or later claim a round contained
something it did not. Every participant simply has to take the server's word for it.

Most systems address one of these. FedVerify addresses all three, and &mdash; this is the
part that matters &mdash; is explicit about where each guarantee stops.

---

## 3. What FedVerify does about each

### Privacy — differential privacy

Before a hospital sends its delta, controlled mathematical noise is added, calibrated so
that whether any single patient was in the training set is provably hard to determine from
the result. The strength is a number called **epsilon (ε)**: lower means more privacy and
less accuracy.

The subtle part, and the thing most implementations get wrong: privacy loss **accumulates**
across rounds. Budgeting for a single round and then running thirty of them silently spends
about thirty times the privacy you claimed. FedVerify solves for the noise level over the
*entire run* up front, and keeps one accountant across all rounds. There is a test whose
only job is to fail if anyone ever changes that.

> **What it does not cover.** The unit of privacy is one training *example*. For MIT-BIH
> that is a single heartbeat, not a patient &mdash; and a patient contributes hundreds of
> beats. A real patient-level guarantee needs different machinery. We say so rather than
> letting the word "private" do the work.

### Poisoning — FedVerify-Forensics

This is the project's own contribution, and it comes from an observation about the existing
BlockVerify system.

BlockVerify already finds a tampered *layer* inside a single model by asking: compared to
all the other layers, does this one look statistically strange? The insight is that **the
exact same question works one level up**. Within a round, instead of comparing D weights
inside one model, compare K client deltas against each other. A malicious participant is an
outlier among its peers in the same way a poisoned layer is an outlier within a model.

So the same detection code is reused &mdash; imported, not rewritten &mdash; and asks four
questions about every client, every round:

| question | catches |
|---|---|
| Is this update far larger or smaller than everyone else's? | scaling attacks, free-riders |
| Does it point in a different direction from the consensus? | sign-flipping |
| Are unusually many individual numbers extreme? | sparse, targeted edits |
| Does it contain NaN, infinity, or absurd values? | corrupted or degenerate updates |

Anything too far out is rejected, and the rest are averaged.

**Why four questions and not one.** A sign-flip attack negates the update. Its *size* is
completely normal &mdash; a size-only detector is blind to it. Only the direction question
sees it. In our runs the size score reads 0.05 while the direction score reads 2022.

**Where the threshold comes from, and why that matters.** There is a cut-off, τ, deciding
how strange is too strange. It is **never chosen by hand.** BlockVerify learned this the
hard way one level down: its original hand-set threshold turned out to flag *half of all
clean layers* on real models. So τ here is derived from data, and every run records the
exact calibration file, key, and policy it came from. The dashboard shows that provenance,
and if a run ever used a hand-typed τ, it says so in amber and warns that the detection
numbers are not meaningful.

*(That warning exists because it caught us. An early demo used a made-up τ of 6.0. The
calibrated value was 2.65 &mdash; our threshold was two and a half times too strict, and
attacker detection sat at 31%. With the calibrated value it went to 75%.)*

### Trust — commitments on a public blockchain

At the end of each round, every client's update is fingerprinted, and all the fingerprints
are combined into one short code called a **Merkle root**. That root is written to a public
blockchain.

Why this is strong:

- Change one number in one client's update &mdash; the root changes completely.
- Swap two clients, or drop one &mdash; the root changes.
- The root is public and permanent, so the server cannot tell two different stories.

Any participant can then ask for a small proof and check, **in their own browser**, that
their update really was in round 7 &mdash; comparing against the root read *straight from
the blockchain*, never from our server. If our server lies, the check fails.

---

## 4. What we built, phase by phase

| phase | what it added |
|---|---|
| **0** | Read the existing codebase and wrote down the ground truth. Found six places where the plan disagreed with reality (wrong App ID, five different Merkle implementations, a wrong test count) and followed the code, not the plan. |
| **1** | The federated training loop: clients, non-IID data splitting, GroupNorm models, reproducible seeding. |
| **2** | Differential privacy with correct across-round accounting, and the privacy–utility experiment. |
| **3** | Round commitments, Merkle proofs, blockchain anchoring, and the five API routes. |
| **4** | The attacks, the robust-aggregation baselines, FedVerify-Forensics, and threshold calibration. |
| **5** | Real hospital data (MIT-BIH heart arrhythmia), split by patient, plus the heterogeneity and DP-vs-attack experiments. |
| **6** | The Federated dashboard tab, the figures, and the documentation. |

**Where it stands:** 258 automated tests pass. The privacy–utility results are real and
complete for MNIST. The remaining experiment grids are running.

Every existing part of BlockVerify was left intact &mdash; the changes to `backend/` and
`frontend/` are **additions only, with zero lines deleted**.

---

## 5. Using the dashboard

Open the **Federated** tab in the sidebar and pick a run from the dropdown. Each panel
answers one question.

### The banner: what am I looking at?

Plain English: which clients are malicious, what the attack does, which defence is running,
and the threshold with its provenance. Read this first.

### Accuracy per Round

Is the model still learning while under attack? Rising and levelling off is healthy.

For heart data, watch **macro-F1**, not accuracy. 89% of heartbeats are normal, so a
useless model that says "normal" every single time still scores 89% accuracy &mdash; and
19% macro-F1. Accuracy on imbalanced data flatters bad models.

### Accept / Reject Lineage

The heart of it. One square per client per round:

| colour | meaning |
|---|---|
| **bright green** | a malicious client, correctly rejected |
| **red** | a malicious client that **got through** |
| **amber** | an honest client wrongly excluded |
| **dark green** | an honest client accepted |

Malicious rows are labelled. A perfect run is bright green on those rows and dark green
everywhere else. Underneath, a sentence states the outcome in words.

### Round Commitments

One row per round: the Merkle root (click to copy), how many updates went in, who was
screened out, and the anchoring transaction.

### Verify proof

The trustless check, in four visible steps:

1. The server hands over a claimed proof. **Nothing is trusted yet.**
2. Your browser **recomputes the fingerprint** from the raw data &mdash; it does not accept
   the server's version of it.
3. It folds the proof upward and checks it reaches the round's root.
4. It reads the anchored root **from the public blockchain directly**, and compares.

Three possible verdicts:

- **TRUSTLESS ✓** &mdash; all four passed. Independently proven.
- **SERVER-TRUSTED** &mdash; steps 1–3 passed, but this run was anchored locally, so there
  is no public ledger for step 4. Honest, and clearly labelled.
- **VERIFICATION FAILED** &mdash; something does not match the commitment.

*(Most demo runs show SERVER-TRUSTED, because they used the local chain. Re-run with
`--chain-backend algorand` to see the full result.)*

---

## 6. The demo runs, and what each one teaches

| run | what to notice |
|---|---|
| **clean** | Nobody malicious. ~4% honest exclusions &mdash; the *designed* cost of a threshold tuned to keep false alarms under 5%. |
| **scaling · forensics** | 75% of malicious updates caught. Attackers inflate their update to overpower the average; the size probe sees it. |
| **scaling · fedavg** | The same attack with **no defence**. Nothing screened at all. This is the control. |
| **sign-flip · forensics** | 75% caught, **zero** honest clients excluded. Caught purely by direction &mdash; the size probe is blind here. |
| **label-flip · forensics** | Only ~44% caught. Honest and important: this attacker *genuinely trained*, just on wrong labels, so its update looks legitimate. |
| **backdoor · forensics** | Only ~6% caught &mdash; but attack success is **1%**, so the backdoor failed anyway. The clearest limitation in the system. |
| **scaling under DP (ε=4)** | 100% caught, but 16% of honest clients excluded and accuracy collapses. Privacy noise and anomaly detection interfere with each other. |

That last row is a genuine research finding rather than a demo, and it is being measured
properly in a dedicated experiment.

---

## 7. What we are careful *not* to claim

Being precise here is what makes the rest credible.

- **We prove which updates entered a round. We do not prove the averaging was done
  correctly.** That is a different and much more expensive problem.
- **An accepted update is not certified benign.** The screening is statistical, and a
  patient attacker who keeps its update inside the normal range will pass. Detection is
  best-effort; the *commitment* is not.
- **We do not detect backdoors reliably.** See the demo table. A backdoored update is a
  legitimately-trained update on poisoned data.
- **There is no automatic recovery.** The record identifies a bad round. Nothing rolls it
  back. In our comparison table, FedVerify's "self-healing rollback" column is marked ✗.
- **Our attacks do not know the defence exists.** An adaptive attacker is the natural next
  step and is not implemented.
- **Privacy is per training example, not per patient.**

---

## 8. Why this combination is new

Plenty of work does robust aggregation. Plenty does differential privacy. Plenty writes
model hashes to a blockchain. Two capabilities are almost universally missing:

1. **A participant can verify its own contribution** against a root read from a public,
   permissionless chain &mdash; not a consortium ledger run by the same people running the
   training.
2. **The accept/reject decision is itself part of the public record.**

That second point is the one worth saying out loud. Krum and median-based defences will
also blunt an attack &mdash; but neither can tell you *who* was excluded or *why*. Krum
keeps one client and stays silent about the rest; median makes no client-level decision at
all.

FedVerify produces a per-client, per-round record, and commits it where nobody &mdash;
including us &mdash; can change it afterwards.

> An excluded participant cannot claim they were treated unfairly, and the coordinator
> cannot claim to have excluded someone they did not.

**We do not claim to beat Krum or median on accuracy.** We claim to be competitive on
accuracy *while additionally* producing a verifiable record that they structurally cannot.
Overclaiming the first would be the fastest way to lose a reviewer on the second.
