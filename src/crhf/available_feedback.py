"""Track available-power forecast errors using only feedback available by the issue slot."""

import heapq
import numpy as np
import pandas as pd
from copy import deepcopy


DAY = 86400 * 10**9


NAT = np.iinfo(np.int64).min


def replay(frame, available, facts, *, state=None, pending_feedback=None, asof=None):
    n = len(frame)
    available = np.asarray(available, float)
    scale = frame.capacity_proxy.to_numpy(float)
    if (
        n == 0
        or available.shape != (n, 2)
        or not np.isfinite(available).all()
        or np.any(available < 0)
    ):
        raise ValueError("Invalid available stream")
    times = pd.DatetimeIndex(frame.Timestamp).as_unit("ns").asi8
    targets = pd.DatetimeIndex(frame.target_timestamp).as_unit("ns").asi8
    if frame[["farm", "Timestamp"]].duplicated().any() or np.any(targets <= times):
        raise ValueError("Invalid issue times")
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("Invalid scale")
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
    width = np.empty((n, 2))
    count = np.empty(n, int)
    latest = np.full(n, NAT, np.int64)
    for farm, ids in frame.groupby("farm", sort=True).indices.items():
        ids = np.asarray(ids)
        ids = ids[np.argsort(times[ids], kind="stable")]
        s = states.get(
            farm, dict(clock=int(times[ids[0]]), last_issue=NAT, pending={}, errors=[], latest=NAT)
        )
        if s["clock"] > int(times[ids[0]]) or s["last_issue"] >= int(times[ids[0]]):
            raise ValueError("Overlapping issue stream")
        queue = []

        def schedule(issue, r):
            row = lookup.get((farm, issue))
            if row is None:
                return
            arrival = row.target_available_at.value
            if (
                row.target_timestamp.value != r["target"]
                or arrival < r["target"]
                or arrival < s["clock"]
            ):
                raise ValueError("Invalid feedback time")
            y = row.target_potential
            good = (
                np.isfinite(y)
                and y >= 0
                and pd.notna(row.target_available_inconsistent)
                and not bool(row.target_available_inconsistent)
            )
            heapq.heappush(queue, (arrival, issue, float(y), good))

        for issue, r in s["pending"].items():
            schedule(issue, r)

        def consume(now):
            while queue and queue[0][0] <= now:
                arrival, issue, y, good = heapq.heappop(queue)
                r = s["pending"].pop(issue)
                if good:
                    s["errors"].append(
                        dict(
                            target=r["target"],
                            arrival=arrival,
                            error=abs(y - r["prediction"]) / r["scale"],
                        )
                    )
                    s["latest"] = arrival
            s["errors"] = [r for r in s["errors"] if r["target"] >= now - 7 * DAY]
            s["clock"] = int(now)

        for i in ids:
            now = int(times[i])
            consume(now)
            count[i] = len(s["errors"])
            latest[i] = s["latest"]
            width[i] = (
                np.maximum(
                    np.quantile(np.array([r["error"] for r in s["errors"]]), 0.9, axis=0), 0.005
                )
                if count[i] >= 16
                else np.array([0.1, 0.1])
            )
            s["pending"][now] = dict(
                target=int(targets[i]), prediction=available[i].copy(), scale=float(scale[i])
            )
            s["last_issue"] = now
            schedule(now, s["pending"][now])
        consume(end)
        states[farm] = s
    return dict(
        width=width, count=count, latest=latest.view("datetime64[ns]"), state=dict(farms=states)
    )
