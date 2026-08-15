# Semantic synonym-choice carrier (offline simulation)

Status: offline simulation, not release-qualified findings. Produced
2026-08-14 on the VPS. Deterministic, stdlib-only; no model required.

## Why this exists

GLOSSOPETRAE's headline claim is that word-choice steganography evades
monitors. The overnight program's P2 phase showed small models cannot
*execute* such encoding (acrostic delivery 0/16, 0.5B/0.6B). This channel
isolates the word-choice *mechanism* deterministically so its detection
profile is measurable without a capable sender: each anchor word
(very/big/small/good/bad/said/happy/quick) carries 2 bits by selecting one
of four synonyms, and the payload is framed with the shared STX1 header.

`src/channels/semantic.py` — `SynonymChoiceChannel`, `canonicalize()`
(the blind monitor's view), `SemanticReportChannel` (the write-report
composition with decoy reads).

## Measured properties

| Property | Result |
| --- | --- |
| Round-trip delivery (word-preserving receiver) | 6/6 (HELLO; 180 anchor slots) |
| Canonicalizing / lemmatizing monitor | blind 6/6 — synonyms collapse to anchors |
| Character-channel exhaustive decoder | blind 6/6 — the semantic axis is invisible to character carriers |
| Synonym-surface scanner | J = 1.0 (honest negative) |
| Character density / tool-diversity scanners | J = 0.0 (isolation) |

## Findings

1. **A third, orthogonal taxonomy axis.** Character (Unicode), structural
   (JSON), and now **lexical (word identity)** — the synonym-surface scanner
   is the only detector that separates the family (J = 1.0) while character
   and diversity scanners are fully blind (J = 0.0). The canonicalizing
   monitor (lemma collapse) is the complete defense, exactly parallel to
   canonicalization defeating the structural carriers.

2. **It closes the P2 loop.** The real-model experiment showed the *sender*
   is the bottleneck at small scale; the offline channel shows the
   *mechanism* itself is sound (deterministic delivery) and its detection
   profile is the same arms-race structure as every other carrier: a
   surface scanner catches it, a normalizing monitor is blind. A capable
   sender model (larger than 0.6B) is the missing piece — the natural
   follow-up when a bigger model or API budget is available.

## Limitations

- Fixed 8-anchor vocabulary; synonyms are hand-chosen, so coverage of
  natural text is thin and the cover reads stilted.
- Case handling is best-effort; hyphenated/possessive forms are not
  canonicalized by the scanner.
- Not a novelty claim: word-choice steganography is a documented class; the
  value here is the measured, isolated detection profile.

Raw results: `findings/semantic/results_*.json` (gitignored).
