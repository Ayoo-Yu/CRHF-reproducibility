"""Select, weight and prune local historical examples for ridge regression."""

import numpy as np


HOUR = 3600 * 10**9


DAY = 24 * HOUR


NAT = np.iinfo(np.int64).min


def nearest(query, queries, k=64):
    distance = np.sum((queries - query) ** 2, axis=1)
    idx = np.argsort(distance, kind="stable")[:k]
    d = distance[idx]
    variance = max(float(np.median(d)), 1e-4)
    weight = np.exp(-d / (2 * variance))
    return idx, weight, float(np.sqrt(d[0]))


def empty_bank():
    return dict(
        examples=[], evidence=[], accepted=0, rejected=0, latest_ns=NAT, latest_target_ns=NAT
    )


def prune(bank, now):
    bank["examples"] = [r for r in bank["examples"] if r["target_ns"] >= now - 14 * DAY][-512:]
    bank["evidence"] = [r for r in bank["evidence"] if r["target_ns"] >= now - 14 * DAY][-2048:]
