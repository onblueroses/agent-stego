# Seeded multi-carrier channel hopping

Status: offline simulation, not release-qualified findings. Produced
2026-08-13 on the VPS. Deterministic, stdlib-only; the trace composition uses
the HonestExecutor, not real models.

## The mechanism

A coordination-primitive channel: the payload is framed once, **rotated** by a
seed-derived offset, split into chunks, and each chunk is encoded into a
separate cover document using a carrier chosen by a **seed-derived schedule**
among the seven carriers (zero-width, NBSP, NFD, homoglyph, tag-char,
variation-selector, JSON key-order).

Two seed-derived transforms protect the stream:

1. **Rotation** ensures no single chunk document contains the frame magic —
   a per-document exhaustive monitor is blind by construction (0/10 catch).
2. **Schedule** makes reassembly require knowing which carrier each chunk
   used; the receiver replays it in O(n * c) work.

The seed is the channel key — the same coordination role as the bootstrapped
private languages, applied to carrier choice instead of codebook content.
This is the offline analogue of negotiated protocols and ST3GG's SPECTER
channel hopping, implemented clean-room.

## Measured properties

| Property | Value |
| --- | --- |
| Seeded receiver delivery | 10/10 |
| Naive per-document exhaustive monitor catch | 0/10 |
| Schedule+rotation search space | n=2: 4,312 · n=3: 30,184 · n=4: 211,288 · n=6: 10.4M · n=8: 507M |
| Brute-force recovery (n=2, n=3) | true payload recovered, but **ambiguous**: 2 and 19 valid-looking candidates |
| Single-carrier density detector (invisible-char) | J = 0.75 (diluted) |
| Full-registry content scanner (OR over all carriers) | catch 12/12, FPR 0/24 |

## The two real findings

1. **Ambiguity, not just cost.** Brute-forcing the schedule finds the true
   payload, but zero-filled chunks pass the magic + length check, so several
   candidate schedules look valid and only one carries the real message. The
   frame's magic pins only the first chunk; the rest of the search is
   exponential and ambiguous. An unscheduled monitor cannot tell which
   candidate is real without ground truth.

2. **Hopping dilutes single-carrier statistics but not full-registry
   scanning.** A detector tuned to one carrier class (e.g. invisible-char
   density) catches only the runs that hop through that class — J drops from
   1.0 (single carrier) to 0.75. A monitor that ORs all content scanners
   still catches every run (12/12, FPR 0). Hopping is a *coordination*
   mechanism, not an invisibility mechanism.

Honest negatives recorded: the naive hopping family (write-only turns) is
caught by the tool-diversity scan at J = 1.0; the round-2 decoy lesson
transfers — matching the cover task's call shape is required for surface
stealth.

## Found-and-fixed during development

- A latent `JsonKeyOrderChannel` bug: the final partially-filled object's
  field was left-aligned incorrectly (padding preceded the placed bits),
  which the single-channel unframe happened to mask but any truncation
  exposed. Placed bits are now left-aligned within the field (regression
  test added).
- The tag-char carrier is byte-oriented; the hopping adapter uses nibble
  encoding (two tag chars per byte) so arbitrary bit chunks survive.

Raw results: `findings/hopping/results_*.json`,
`findings/trace_detection/results_*.json` (gitignored).
