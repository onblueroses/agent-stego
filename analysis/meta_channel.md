# Carrier meta-channel: keyless cross-surface hopping

Status: offline simulation, not release-qualified findings. Produced
2026-08-14 on the VPS. Deterministic, stdlib-only.

## The mechanism

`src/channels/meta_hop.py` — round-4 hopping needs a shared seed; this
channel removes the seed. Chunk 0 is always a structural (key-order) config
carrying a header that **names the carriers** for the remaining chunks; the
payload chunks ride the named text carriers. The receiver decodes chunk 0,
reads the schedule, then decodes the payload. No shared key, no side
information — the coordination key travels with the payload, on a *different
surface* than the payload itself.

## Measured properties

| Property | Result |
| --- | --- |
| Joint decode (keyless) | 3/3 — no seed, the schedule is embedded |
| Text-surface monitor: true payload recovery | 0/6 — never the message |
| Text-surface monitor: ambiguous garbage frames | 1/6 — chunk 1 on substitution carriers |
| Structural-surface monitor | schedule-only 3/3 — header, no payload |
| Canonicalized header | kills the whole channel |
| Key-order scanner | J = 1.0 (honest negative) |

## Findings

1. **Keyless coordination.** The schedule rides in a structural chunk while
   the payload rides in character carriers, so a text-carrier monitor never
   reads the schedule and a structural-only monitor reads the schedule but
   no payload. Only the joint decoder recovers the message — no shared
   secret anywhere.

2. **The ambiguity structure persists, localized.** Chunk 1 carries the
   frame magic+length; on *substitution-based* carriers (homoglyph, NBSP,
   NFD) the cover's unmodified positions complete a zero-filled garbage
   frame — the round-4 ambiguity, now localized to the first chunk.
   *Insertion* carriers (zero-width, variation-selector) and tag-char are
   fully blind. The text surface never recovers the true payload.

3. **Single point of failure.** Canonicalizing the structural surface
   destroys the schedule and therefore the whole channel — the price of
   keylessness vs round-4's distributed seed.

## Limitations

- The bootstrap assumption (chunk 0 is key-order) is protocol-fixed, like a
  default port.
- Substitution carriers on chunk 1 leak ambiguous frames; insertion
  carriers avoid this.

Raw results: `findings/meta_hop/results_*.json` (gitignored).
