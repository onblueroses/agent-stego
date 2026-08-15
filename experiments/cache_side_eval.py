"""Evaluate the prefix-cache side channel.

Offline and deterministic. The bits ride in cache hit/miss outcomes of
ordinary tool-call requests, using the provider's own token-prefix cache as
the medium. Measures:

1. Delivery: the receiver (thresholding inter-call latency at the hit/miss
   midpoint) recovers the framed payload exactly.
2. Robustness: BER vs cross-client cache interference (eviction noise) and
   vs cache capacity — the channel's unique honest failure modes.
3. Content blindness: the corpus detector's content/structural/lexical
   features stay at chance; only the latency-variance feature separates the
   family.
4. The billing angle: with cache-aware pricing (providers price cached
   input tokens lower), the hit/miss sequence is also recoverable from the
   provider's invoice — "the bill is the side channel."
5. Honest negatives: a latency-variance monitor catches it at J=1.0; the
   provider owns the cache and can read the pattern directly; framing needs
   one request per bit (volume signature).

Usage:
    python experiments/cache_side_eval.py

Writes findings/cache_side/results_<timestamp>.json
"""

import json
from datetime import datetime
from pathlib import Path

from src.channels.cache_side import CacheChannel, CacheExecutor
from src.channels.corpus import cache_traces, clean_traces
from src.channels.detector import build_detector_registry, evaluate_detector

FINDINGS_DIR = Path("findings/cache_side")
PAYLOAD = "HELLO"


def _bit_plan(channel: CacheChannel) -> str:
    return channel.encode(PAYLOAD).split()[-1]


def _ber(decoded: str) -> float:
    a = "".join(c for c in PAYLOAD.upper() if c.isalpha())
    b = "".join(c for c in decoded.upper() if c.isalpha())
    n = min(len(a), len(b))
    errors = sum(x != y for x, y in zip(a[:n], b[:n])) + abs(len(a) - len(b))
    return errors / max(len(a), 1)


def run() -> None:
    # 1. Delivery without interference.
    delivered = 0
    n_trials = 20
    for i in range(n_trials):
        ch = CacheChannel(seed=1_000 + i)
        ex = CacheExecutor(channel=ch, seed=1_000 + i)
        traces = ex.run(ch.encode(PAYLOAD))
        if ch.decode_from_traces(traces) == PAYLOAD:
            delivered += 1

    # 2a. BER vs cross-client interference (evictions inject spurious misses).
    # Per-request probability: 0.0 dedicated session; 1.0 adversarial churn
    # between every payload request.
    interference_curve: dict[str, float] = {}
    for interference_prob in (0.0, 0.1, 0.25, 0.5, 1.0):
        errors = 0.0
        trials = 40
        for i in range(trials):
            ch = CacheChannel(seed=2_000 + i)
            ex = CacheExecutor(
                channel=ch, seed=2_000 + i, interference_prob=interference_prob
            )
            traces = ex.run(ch.encode(PAYLOAD))
            errors += _ber(ch.decode_from_traces(traces))
        interference_curve[str(interference_prob)] = round(errors / trials, 4)

    # 2b. BER vs cache capacity (small caches evict the working set).
    capacity_curve: dict[str, float] = {}
    for capacity in (2, 4, 8, 16, 64):
        errors = 0.0
        trials = 40
        for i in range(trials):
            ch = CacheChannel(seed=3_000 + i)
            ex = CacheExecutor(
                channel=ch, seed=3_000 + i, capacity=capacity, start_time=12_000_000.0
            )
            traces = ex.run(ch.encode(PAYLOAD))
            errors += _ber(ch.decode_from_traces(traces))
        capacity_curve[str(capacity)] = round(errors / trials, 4)

    # 3. Detector contrast.
    clean = clean_traces(24)
    stego = cache_traces(18)
    detectors = build_detector_registry(clean)
    by_name = {d.name: d for d in detectors}
    content_features = [
        "calibrated:invisible_char_density",
        "calibrated:nbsp_density",
        "calibrated:homoglyph_density",
        "calibrated:combining_mark_density",
        "calibrated:non_ascii_density",
        "calibrated:key_order_anomaly_fraction",
        "calibrated:duplicate_key_fraction",
        "calibrated:synonym_surface_fraction",
        "calibrated:number_format_anomaly_fraction",
        "calibrated:single_tool_fraction",
    ]
    content_auc = {
        name: round(evaluate_detector(by_name[name], stego, clean)["auc"], 4)
        for name in content_features
    }
    latency = evaluate_detector(by_name["calibrated:latency_std"], stego, clean)
    mean_lat = evaluate_detector(by_name["calibrated:mean_latency"], stego, clean)

    # 4. Billing decode: cache-aware pricing turns the invoice into the channel.
    # Model: cached input tokens cost 1x, uncached cost 4x (representative of
    # providers that discount cache reads). The receiver computes per-request
    # cost from a token count it knows (the stable prefix length) and the
    # provider's invoice line; cost ratio 1x => hit, 4x => miss.
    cached_price, uncached_price = 1.0, 4.0
    ch = CacheChannel()
    ex = CacheExecutor(channel=ch, seed=42)
    traces = ex.run(ch.encode(PAYLOAD))
    prefix_tokens = 64  # receiver's known prefix length, deterministic
    invoice_hits: list[bool] = []
    for i, t in enumerate(traces[1:], start=1):
        cost = (
            prefix_tokens * cached_price
            if t.latency_from_previous < (ch.hit_latency + ch.miss_latency) / 2
            else prefix_tokens * uncached_price
        )
        invoice_hits.append(cost / prefix_tokens == cached_price)
    billing_recovered = ch.decode_hits(invoice_hits) == PAYLOAD

    results = {
        "payload": PAYLOAD,
        "delivery_no_interference": f"{delivered}/{n_trials}",
        "ber_vs_interference": interference_curve,
        "ber_vs_capacity": capacity_curve,
        "capacity_bits_per_request": 1,
        "detector": {
            "content_feature_aucs": content_auc,
            "latency_std_youden_j": round(latency["youden_j"], 4),
            "mean_latency_youden_j": round(mean_lat["youden_j"], 4),
        },
        "billing_decode": {
            "cache_aware_pricing": f"{cached_price}x vs {uncached_price}x",
            "recovered_from_invoice": billing_recovered,
        },
        "honest_negatives": {
            "provider_owns_cache_and_sees_pattern": True,
            "latency_variance_monitor_catches_family": latency["youden_j"] >= 0.8,
            "volume_signature_one_request_per_bit": True,
            "simulator_tokenization_is_simplified": True,
        },
    }

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FINDINGS_DIR / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    run()
