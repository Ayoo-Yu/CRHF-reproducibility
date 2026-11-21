"""Select corrected or base forecasts from prior, version-matched gains in each risk regime."""

import heapq
import numpy as np
import pandas as pd
from copy import deepcopy
from .roles import CAP


DAY = 86400 * 10**9


NAT = np.iinfo(np.int64).min


def adjust_daily(state, record, sign):
    key = (record["context"], record["version"], record["target_ns"] // DAY)
    stat = state["daily"].setdefault(
        key, dict(total=np.zeros((2, 3)), count=np.zeros((2, 3), np.int64))
    )
    stat["total"] += sign * record["gain"] * record["executed"]
    stat["count"] += sign * record["executed"]
    if not stat["count"].any():
        del state["daily"][key]


def prune(state, now):
    keep = []
    for r in state["records"]:
        if r["target_ns"] < now - 14 * DAY:
            adjust_daily(state, r, -1)
        else:
            keep.append(r)
    state["records"] = keep


def qualification(daily, context, version, bank, match_version):
    byday = {}
    for (ctx, ver, day), stats in daily.items():
        if ctx != context or (match_version and ver != version):
            continue
        d = byday.setdefault(day, [np.zeros(3), np.zeros(3, np.int64)])
        d[0] += stats["total"][bank]
        d[1] += stats["count"][bank]
    score = np.array([0.0, -1.0, -1.0])
    days = np.zeros(3, int)
    counts = np.zeros(3, int)
    if not byday:
        return score, days, counts
    total = np.array([v[0] for v in byday.values()])
    count = np.array([v[1] for v in byday.values()])
    for j in (1, 2):
        valid = count[:, j] >= 8
        days[j] = int(valid.sum())
        counts[j] = int(count[valid, j].sum())
        if days[j] < 3 or counts[j] < 32:
            continue
        values = total[valid, j] / count[valid, j]
        score[j] = float(values.mean() - 1.64 * values.std(ddof=1) / np.sqrt(len(values)))
    return score, days, counts


def choose(score):
    largest = float(np.max(score))
    return int(np.flatnonzero(score >= largest - 1e-12)[0]) if largest > 0.0001 else 0


def replay(
    frame,
    incumbent,
    plans,
    executed,
    contexts,
    versions,
    facts,
    *,
    state=None,
    pending_feedback=None,
    asof=None,
):
    n = len(frame)
    incumbent = np.asarray(incumbent, float)
    plans = np.asarray(plans, float)
    executed = np.asarray(executed, bool)
    contexts = np.asarray(contexts, int)
    versions = np.asarray(versions, str)
    scale = frame.capacity_proxy.to_numpy(float)
    times = pd.DatetimeIndex(frame.Timestamp).as_unit("ns").asi8
    targets = pd.DatetimeIndex(frame.target_timestamp).as_unit("ns").asi8
    if n == 0 or frame[["farm", "Timestamp"]].duplicated().any() or np.any(targets <= times):
        raise ValueError("Invalid issue chronology")
    if (
        incumbent.shape != (n,)
        or plans.shape != (n, 2, 3)
        or executed.shape != plans.shape
        or contexts.shape != (n,)
        or versions.shape != (n,)
    ):
        raise ValueError("Invalid dimensions")
    if (
        not np.isfinite(plans).all()
        or np.any(plans < 0)
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("Invalid plans/scales")
    np.testing.assert_allclose(
        plans[:, :, 0], np.repeat(incumbent[:, None], 2, axis=1), atol=1e-12, rtol=0
    )
    if np.any(abs(plans - incumbent[:, None, None]) > (CAP + 1e-9) * scale[:, None, None]):
        raise ValueError("Tool exceeds correction bound")
    if np.any((contexts < 0) | (contexts > 3)) or not executed[:, :, 0].all():
        raise ValueError("Invalid context/fallback")
    end = pd.Timestamp(frame.Timestamp.max() if asof is None else asof).value
    if end < times.max():
        raise ValueError("Backdated asof")
    ff = (
        facts
        if pending_feedback is None
        else pd.concat([pending_feedback, facts], ignore_index=True)
    )
    lookup = {
        (r.farm, r.Timestamp.value): r
        for r in ff.drop_duplicates(["farm", "Timestamp"], keep="last").itertuples(index=False)
    }
    states = {} if state is None else deepcopy(state["farms"])
    out = dict(
        predictions=np.empty((n, 3)),
        actions=np.empty((n, 3), int),
        scores=np.empty((n, 3, 3)),
        days=np.empty((n, 3, 3), int),
        counts=np.empty((n, 3, 3), int),
        latest=np.full(n, NAT, np.int64),
        accepted=np.zeros(n, int),
    )
    for farm, ids in frame.groupby("farm", sort=True).indices.items():
        ids = np.asarray(ids)
        ids = ids[np.argsort(times[ids], kind="stable")]
        s = states.get(
            farm,
            dict(
                clock_ns=int(times[ids[0]]),
                last_issue_ns=NAT,
                pending={},
                records=[],
                daily={},
                accepted=0,
                latest_ns=NAT,
            ),
        )
        if s["clock_ns"] > int(times[ids[0]]) or s["last_issue_ns"] >= int(times[ids[0]]):
            raise ValueError("Overlapping issue stream")
        queue = []

        def schedule(issue, r):
            row = lookup.get((farm, issue))
            if row is None:
                return
            at = row.target_available_at.value
            if (
                row.target_timestamp.value != r["target_ns"]
                or at < r["target_ns"]
                or at < s["clock_ns"]
            ):
                raise ValueError("Invalid feedback timing")
            y = row.target_actual_operational
            good = (
                np.isfinite(y)
                and y >= 0
                and pd.notna(row.target_available_inconsistent)
                and not bool(row.target_available_inconsistent)
            )
            heapq.heappush(queue, (at, issue, float(y), good))

        for issue, r in s["pending"].items():
            schedule(issue, r)

        def consume(now):
            while queue and queue[0][0] <= now:
                arrival, issue, y, good = heapq.heappop(queue)
                r = s["pending"].pop(issue)
                if good:
                    gain = (abs(y - r["incumbent"]) - abs(y - r["plans"])) / r["scale"]
                    record = dict(
                        issue_ns=issue,
                        target_ns=r["target_ns"],
                        arrival_ns=arrival,
                        context=r["context"],
                        version=r["version"],
                        gain=gain,
                        executed=r["executed"],
                    )
                    s["records"].append(record)
                    adjust_daily(s, record, 1)
                    s["accepted"] += 1
                    s["latest_ns"] = arrival
            prune(s, now)
            s["clock_ns"] = int(now)

        for i in ids:
            now = int(times[i])
            consume(now)
            for p, (bank, match) in enumerate(((0, True), (0, False), (1, True))):
                score, days, counts = qualification(
                    s["daily"], int(contexts[i]), str(versions[i]), bank, match
                )
                score[~executed[i, bank]] = -1.0
                score[0] = 0.0
                action = choose(score)
                out["predictions"][i, p] = plans[i, bank, action]
                out["actions"][i, p] = action
                out["scores"][i, p] = score
                out["days"][i, p] = days
                out["counts"][i, p] = counts
            out["latest"][i] = s["latest_ns"]
            out["accepted"][i] = s["accepted"]
            s["pending"][now] = dict(
                target_ns=int(targets[i]),
                incumbent=float(incumbent[i]),
                scale=float(scale[i]),
                plans=plans[i].copy(),
                executed=executed[i].copy(),
                context=int(contexts[i]),
                version=str(versions[i]),
            )
            s["last_issue_ns"] = now
            schedule(now, s["pending"][now])
        consume(end)
        states[farm] = s
    out["latest"] = out["latest"].view("datetime64[ns]")
    out["state"] = dict(farms=states)
    return out
