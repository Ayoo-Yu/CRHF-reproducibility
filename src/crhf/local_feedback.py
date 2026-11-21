"""Maintain arrived residuals and apply qualified neighborhood ridge corrections to direct CatBoost."""

import heapq
import numpy as np
import pandas as pd
from copy import deepcopy
from .neighborhood import empty_bank, nearest, prune


HOUR = 3600 * 10**9


DAY = 24 * HOUR


NAT = np.iinfo(np.int64).min


CAP = 0.05


def execute(anchor, scale, query, examples):
    correction = 0.0
    executed = np.array([True, False])
    if len(examples) >= 16:
        q = np.array([r["query"] for r in examples])
        y = np.array([r["error"] for r in examples])
        ix, w, d = nearest(query, q)
        if d <= 1.5:
            z = np.c_[np.ones(len(ix)), q[ix] - query]
            beta = np.linalg.solve(
                z.T @ (w[:, None] * z) + np.diag(np.r_[4.0, np.full(12, 16.0)]), z.T @ (w * y[ix])
            )
            correction = beta[0]
            executed[1] = True
    plans = np.maximum(anchor + scale * np.clip(np.array([0.0, correction]), -CAP, CAP), 0.0)
    return plans, executed


def gains(query, bank, now, executed):
    score = np.array([0.0, -1.0])
    evidence = bank["evidence"]
    if not executed[1] or not evidence:
        return score
    records = [r for r in evidence if now - 14 * DAY <= r["target_ns"] <= now and r["executed"][1]]
    if len(records) < 16:
        return score
    q = np.array([r["query"] for r in records])
    ix, w, d = nearest(query, q)
    if w.sum() ** 2 / np.dot(w, w) < 16 or d > 1.5:
        return score
    y = np.array([records[i]["gain"][1] for i in ix])
    score[1] = np.sum(w * y) / w.sum()
    return score


def choose(score):
    v = np.asarray(score, float)
    if v.shape[-1] != 2 or not np.isfinite(v).all():
        raise ValueError("Invalid two-tool values")
    maximum = v.max(axis=-1)
    tied = v >= maximum[..., None] - 1e-12
    return np.where(maximum > 0.0001, tied.argmax(axis=-1), 0).astype(int)


def learn(bank, record, y, arrival):
    error = (y - record["anchor"]) / record["scale"]
    gain = (abs(record["anchor"] - y) - abs(record["plans"] - y)) / record["scale"]
    common = dict(
        issue_ns=record["issue_ns"],
        target_ns=record["target_ns"],
        arrival_ns=int(arrival),
        query=record["query"].copy(),
    )
    bank["evidence"].append(dict(**common, gain=gain, executed=record["executed"].copy()))
    bank["evidence"].sort(key=lambda r: (r["target_ns"], r["issue_ns"]))
    bank["evidence"] = bank["evidence"][-2048:]
    if record["target_ns"] % HOUR == 0:
        bank["examples"].append(dict(**common, error=float(error)))
        bank["examples"].sort(key=lambda r: (r["target_ns"], r["issue_ns"]))
        bank["examples"] = bank["examples"][-512:]
    bank["accepted"] += 1
    bank["latest_ns"] = int(arrival)
    bank["latest_target_ns"] = max(bank["latest_target_ns"], record["target_ns"])


def replay(frame, anchors, queries, feedback, *, state=None, pending_feedback=None, asof=None):
    n = len(frame)
    anchors = np.asarray(anchors, float)
    queries = np.asarray(queries, float)
    times = pd.DatetimeIndex(frame.Timestamp).as_unit("ns").asi8
    targets = pd.DatetimeIndex(frame.target_timestamp).as_unit("ns").asi8
    if n == 0 or frame[["farm", "Timestamp"]].duplicated().any() or np.any(targets <= times):
        raise ValueError("Invalid issue stream")
    if (
        anchors.shape != (n,)
        or queries.shape != (n, 12)
        or not np.isfinite(queries).all()
        or not np.isfinite(anchors).all()
        or np.any(anchors < 0)
    ):
        raise ValueError("Invalid query/anchor")
    scales = frame.capacity_proxy.to_numpy(float)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Invalid scale")
    end = pd.Timestamp(frame.Timestamp.max() if asof is None else asof).value
    if end < times.max():
        raise ValueError("asof before last issue")
    facts = (
        feedback
        if pending_feedback is None
        else pd.concat([pending_feedback, feedback], ignore_index=True)
    )
    facts = facts.drop_duplicates(["farm", "Timestamp"], keep="last")
    lookup = {(r.farm, r.Timestamp.value): r for r in facts.itertuples(index=False)}
    states = {} if state is None else deepcopy(state["farms"])
    out = dict(
        predictions=np.empty(n),
        actions=np.empty(n, int),
        tools=np.empty((n, 2)),
        scores=np.empty((n, 2)),
        executed=np.empty((n, 2), bool),
        latest=np.full(n, NAT, np.int64),
        accepted=np.zeros(n, int),
        examples=np.zeros(n, int),
    )
    for farm, ids in frame.groupby("farm", sort=True).indices.items():
        ids = np.asarray(ids)
        ids = ids[np.argsort(times[ids], kind="stable")]
        s = states.get(
            farm,
            dict(clock_ns=int(times[ids[0]]), last_issue_ns=NAT, normal=empty_bank(), pending={}),
        )
        if s["last_issue_ns"] >= int(times[ids[0]]) or s["clock_ns"] > int(times[ids[0]]):
            raise ValueError("Overlapping/backdated issue")
        queue = []

        def schedule(issue, record):
            row = lookup.get((farm, issue))
            if row is None:
                return
            at = row.target_available_at.value
            if (
                row.target_timestamp.value != record["target_ns"]
                or at < record["target_ns"]
                or at < s["clock_ns"]
            ):
                raise ValueError("Backdated/mismatched outcome")
            y = row.target_actual_operational
            valid = (
                np.isfinite(y)
                and y >= 0
                and pd.notna(row.target_available_inconsistent)
                and not bool(row.target_available_inconsistent)
            )
            heapq.heappush(queue, (at, issue, float(y), valid))

        for issue, r in s["pending"].items():
            schedule(issue, r)

        def consume(now):
            while queue and queue[0][0] <= now:
                arrival, issue, y, valid = heapq.heappop(queue)
                record = s["pending"].pop(issue)
                prune(s["normal"], arrival)
                if valid:
                    learn(s["normal"], record, y, arrival)
            prune(s["normal"], now)
            s["clock_ns"] = int(now)

        for i in ids:
            now = int(times[i])
            consume(now)
            bank = s["normal"]
            plans, executed = execute(
                float(anchors[i]), float(scales[i]), queries[i], bank["examples"]
            )
            score = gains(queries[i], bank, now, executed)
            action = choose(score)
            out["predictions"][i] = plans[action]
            out["actions"][i] = action
            out["tools"][i], out["scores"][i], out["executed"][i] = plans, score, executed
            out["latest"][i], out["accepted"][i], out["examples"][i] = (
                bank["latest_ns"],
                bank["accepted"],
                len(bank["examples"]),
            )
            record = dict(
                issue_ns=now,
                target_ns=int(targets[i]),
                query=queries[i].copy(),
                anchor=float(anchors[i]),
                scale=float(scales[i]),
                plans=plans,
                executed=executed,
            )
            s["pending"][now] = record
            s["last_issue_ns"] = now
            schedule(now, record)
        consume(end)
        states[farm] = s
    out["latest"] = out["latest"].view("datetime64[ns]")
    out["state"] = dict(farms=states)
    return out
