# Spread-spectrum multi-carrier channel

Status: offline simulation, not release-qualified findings. Produced
2026-08-14 on the VPS. Deterministic, stdlib-only.

## The mechanism

`src/channels/spread.py` — real monitoring stacks canonicalize SPECIFIC
axes: NFC normalization kills NFD, whitespace normalization kills NBSP,
invisible-character stripping kills zero-width/tag-char. This channel
replicates every bit across several carriers, so stripping any *single*
carrier class no longer kills the message: the receiver tries each
(carrier, document) pair and takes the first that unframes.

## Measured properties

| Property | Result |
| --- | --- |
| Round-trip delivery | 6/6 (HELLO replicated over zero-width + NBSP + homoglyph) |
| Strip ONE carrier (canonicalizing monitor) | 6/6 still deliver — for each of the three carriers |
| Strip ALL carriers | 0/6 — dead |
| Detection (honest negative) | invisible-char, NBSP, AND homoglyph scanners all J = 1.0 |

## Findings

1. **Single-axis canonicalization is no longer a defense.** A monitoring
   pipeline that normalizes one Unicode axis (the common case) strips only
   that carrier; the replicated copies in the other carriers still deliver.
   This hardens the character-carrier family against the exact
   normalization defenses that defeated every single-carrier channel in the
   earlier rounds.

2. **Robustness is purchased with detectability.** Redundancy multiplies the
   statistical signal — three scanners fire at J = 1.0 (vs one for a
   single-carrier channel). The tradeoff mirrors margin in the timing
   channel: robustness costs bandwidth/stealth, here in the form of
   multiplied signal.

3. **The defense is to strip ALL axes or detect the redundancy.** A
   pipeline that canonicalizes every Unicode axis kills it; a full-registry
   scanner catches the multiplied signal trivially.

Raw results: `findings/spread/results_*.json` (gitignored).
