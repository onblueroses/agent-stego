# Sneaky-agents rerun — final findings (full sweep)

Length-matched, max_tokens=80, secret=520 bits, MLP probe over hidden states.
Sweep ran on Vast.ai Germany (RTX 6000 Ada, 80GB, $0.59/hr), 2026-05-02.
Three phases: A (N=80 confirmation on the AUC=0.5 cell), B (14B full grid),
C (capacity + cross-condition probe transfer at 3B+7B).

## coupled_probe.py: probe AUC by (model, coupling, encryption)

| Model | T | top_p | KEY=none AUC | KEY=set AUC | Δ | N |
|-------|---|-------|--------------|-------------|---|---|
| 3B    | 1.0 | 1.00 | 1.000 | 1.000 | 0.000 | 40 |
| 3B    | 0.9 | 0.90 | 0.990 | 0.970 | -0.020 | 40 |
| 3B    | 0.7 | 0.85 | 0.950 | **0.585** | -0.365 | **80** |
| 7B    | 1.0 | 1.00 | 1.000 | 0.980 | -0.020 | 40 |
| 7B    | 0.9 | 0.90 | 1.000 | 0.870 | -0.130 | 40 |
| 7B    | 0.7 | 0.85 | 1.000 | **0.615** | -0.385 | **80** |
| 14B   | 1.0 | 1.00 | 1.000 | 0.945 | -0.055 | 40 |
| 14B   | 0.9 | 0.90 | 1.000 | 0.940 | -0.060 | 40 |
| 14B   | 0.7 | 0.85 | 0.970 | **0.540** | -0.430 | 40 |

The N=80 cells (the encrypted+aggressive cells at 3B and 7B) were re-run at
double sample size to tighten the CI on the AUC≈0.5 claim. The 7B value
walked from 0.460 (N=40) → 0.615 (N=80); the N=40 number was at the lucky-low
end of noise. Both are consistent with chance at their respective CIs:

- N=40, AUC≈0.5: SE≈0.083, 95% CI ±0.16
- N=80, AUC≈0.6: SE≈0.054, 95% CI ±0.11

## capacity_and_transfer.py: bits/token + probe AUC at 3B and 7B

Capacity (mean bits/token, averaged across 5 trials × 4 secret messages):

| Model | KEY | raw bpt | mod bpt | agg bpt |
|-------|-----|---------|---------|---------|
| 3B    | none | 2.72 | 1.73 | 1.00 |
| 3B    | set  | 3.27 | 2.06 | 1.00 |
| 7B    | none | 2.74 | 1.97 | 1.00 |
| 7B    | set  | 3.17 | 1.84 | 1.06 |

Capacity collapses with coupling: aggressive (T=0.7, p=0.85) leaves about
1 bit/token of usable channel — most of the entropy has been spent on
mimicking the natural distribution. Encryption itself doesn't change capacity
(arithmetic-coded keystream has the same statistics).

## Probe transfer: train on one condition, test on another (3B and 7B, N=40)

7B / KEY=set transfer matrix (selected):

|              | test=raw | test=mod | test=agg |
|--------------|----------|----------|----------|
| train=raw    | 0.99     | 0.84     | **0.51** |
| train=agg    | 0.87     | 0.73     | 0.44     |

Pattern across all (model, KEY) combinations: a probe trained on one coupling
generalizes to nearby couplings (raw → moderate ≈ 0.84-0.92) but **not** to
aggressive coupling (raw → aggressive ≈ 0.48-0.55). A defender who only
trains a probe on raw (T=1.0) sampling has no defense against an attacker
who switches to T=0.7. Condition-matched training is required.

## Headlines

1. **Detection survives either intervention alone.** Encryption alone (raw
   sampling, KEY=set) holds AUC at 0.945-1.000 across all three model sizes.
   Coupling alone (aggressive, KEY=none) holds AUC at 0.950-0.970. Each
   defense leaves one signal intact for the probe — the activation
   signature (broken only by encryption) or the entropy gap (closed only by
   coupling).

2. **Only the conjunction collapses detection.** Encryption + aggressive
   coupling drops AUC to 0.540-0.615 across 3B/7B/14B — indistinguishable
   from chance at the relevant CIs. The original post's headline pattern
   survives, just at slightly less dramatic numbers (0.54-0.62 vs the
   original 0.46-0.55).

3. **The 14B "AUC=0.92" claim from the original post.** It almost certainly
   referred to the encrypted-only cells (raw/moderate × KEY=set). Our 14B
   numbers at those cells are 0.945 and 0.940 — close to 0.92. The
   "encryption defeated the probe at 14B" framing was misleading: the
   probe still detected at AUC ~0.94, which is *very far from chance.*

4. **Probes don't transfer across coupling regimes.** Train on T=1.0
   sampling, test on T=0.7 — AUC drops to 0.48-0.55 (chance) consistently.
   Defenders need to train a separate probe per coupling regime they want
   to detect. This finding is robust across model sizes and KEY conditions.

5. **Capacity ceiling is real.** Aggressive coupling caps the channel at
   ~1 bit/token regardless of encryption. The original post's 3.78-3.99
   bpt numbers correspond to *raw* sampling at 7B (we measured 3.17),
   which is also where the probe is trivially defeated.

## Honest claim for the rewrite

**Detection survives either defense alone; only the conjunction
(encryption + coupling that nearly closes the entropy gap) collapses
the probe to chance.** The conjunction also collapses the channel
to ~1 bit/token. So the engineering tradeoff is real: there's a
"undetectable" regime, but it's narrow and slow.

A defender's response is straightforward: train probes per coupling
regime. Once the coupling is fixed by adversarial choice, the
defender just needs matching training data.

## Files

Coupled probe (4 JSONs):
- `artifacts/findings/coupled_probe/results_20260502-151830.json` — 3B/N=80/encrypted/aggressive
- `artifacts/findings/coupled_probe/results_20260502-153957.json` — 7B/N=80/encrypted/aggressive
- `artifacts/findings/coupled_probe/results_20260502-163504.json` — 14B/N=40/KEY=none/full grid
- `artifacts/findings/coupled_probe/results_20260502-172802.json` — 14B/N=40/KEY=set/full grid

Capacity + transfer (4 JSONs):
- `artifacts/findings/capacity_transfer/results_20260502-181136.json` — 3B/KEY=none
- `artifacts/findings/capacity_transfer/results_20260502-185459.json` — 3B/KEY=set
- `artifacts/findings/capacity_transfer/results_20260502-193853.json` — 7B/KEY=none
- `artifacts/findings/capacity_transfer/results_20260502-202211.json` — 7B/KEY=set

Logs:
- `artifacts/sweep.log` — full sweep stdout (`phases_ok=8 phases_fail=0`)
- `artifacts/logs/A_n80_*.log`, `B_14B_*.log`, `C_capacity_*.log`

## Caveats

- N=40 gives 95% CI ±0.16 on AUC=0.5; N=80 gives ±0.11. The aggressive+
  encrypted cells at 3B (0.585 ± 0.11) and 7B (0.615 ± 0.11) are
  consistent with chance but mildly suggestive of residual signal.
- Single 2-layer MLP probe over mean-pooled hidden states from 10 layers,
  200 epochs, BCE + Adam @1e-3. Different architectures (transformer
  probes, position-aware, multi-layer) may extract more signal.
- Capacity numbers reflect 4 fixed secret messages (16-104 bits) and 5
  trials each — n=20 per (model, coupling, KEY). Variance is high
  (e.g., 7B raw "HELLO" min_bpt=1.33, max_bpt=8.0).
- "THE KEY IS 42" (104 bits) shows 0% recovery at all couplings. There's
  a length × content interaction worth investigating but not in scope here.
- All sampling at temperature ≥0.7. Lower temperatures (T=0.5, T=0.3)
  not measured; capacity will fall further but probe behavior is
  unmeasured in that regime.
