"""Fit CRHF chronologically, export checkpoints, and compare independent predictions."""

import time

import joblib
import numpy as np
import pandas as pd

from copy import deepcopy
from .local_feedback import replay as direct_replay
from .spatial import spatial_features
from .backbones import fit_cat, predict_cat
from .protocol import FOLDS, month_mask, training_mask
from .interfaces import query
from .correction_selection import replay as actor_replay
from .roles import STARTS, versions
from .residual_models import contexts, features, fit_tools, history_columns
from .fold_inference import phase_plans
from .available_feedback import replay as quality_replay
from .risk_bridge import build_risk_stream, predict_risk
from .pipeline import (
    KEYS,
    POLICIES,
    ROLES,
    compare,
    eligible,
    feedback,
    public_keys,
    reconstruct,
    static_frame,
    write_json,
    write_metrics,
)
from .available_residual import fit as fit_center, predict as predict_center


def train(out):
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(2)
    started = time.monotonic()
    receipts = []
    d = reconstruct()
    f = d["frame"]
    cfg = d["config"]
    schema = d["schema"]

    def note(stage, **data):
        rec = dict(stage=stage, elapsed_seconds=round(time.monotonic() - started, 3), **data)
        receipts.append(rec)
        write_json(out / "training_receipts.json", receipts)
        print(rec, flush=True)

    note("raw_data", rows=len(f))
    wf = d["all_frame"]
    parts = [wf.loc[wf.phase.eq(v)].copy() for v in ("train", "validation", "test")]
    with threadpool_limits(limits=8):
        _, _, _, bridges, risk_meta = build_risk_stream(*parts, d["risk_spec"], cfg, 4, print)
    f["risk_q"] = np.nan
    f["risk_raw_f19"] = np.nan
    f["risk_fold_start"] = pd.NaT
    for start in FOLDS:
        hit = month_mask(f, start)
        b = bridges[str(pd.Timestamp(start))] if start < "1997-12-01" else bridges["final"]
        f.loc[hit, "risk_raw_f19"], f.loc[hit, "risk_q"] = predict_risk(
            b, static_frame(f.loc[hit])
        )
        f.loc[hit, "risk_fold_start"] = pd.Timestamp(b["cutoff"])
    joblib.dump(bridges, out / "stage1_bridges.joblib", compress=3)
    write_json(out / "stage1_training.json", risk_meta)
    note("stage1", risk_bridges=len(bridges))
    sp = spatial_features(d["source"], f).astype(np.float32)
    assert list(sp) == schema["spatial_columns"]
    scale = f.capacity_proxy.to_numpy(float)
    models = {}
    centers = {}
    av = f[KEYS].copy()
    anchor = np.full(len(f), np.nan)
    for task in ("available", "direct"):
        if task == "available":
            names = schema["columns"]["spatial"]
            x = pd.concat([f[schema["columns"]["aggregate"]], sp], axis=1)[names].to_numpy(
                np.float32
            )
            current = f.potential.to_numpy(float)
            y = (f.target_potential.to_numpy(float) - current) / scale
            for v in ("raw", "full", "current"):
                av["spatial_" + v] = np.nan
        else:
            names = [f"seq_{i}" for i in range(576)] + schema["observed"] + list(sp)
            x = np.c_[
                d["sequence"].reshape(len(f), -1),
                f[schema["observed"]].to_numpy(np.float32),
                sp.to_numpy(np.float32),
            ]
            current = f.actual_operational.to_numpy(float)
            y = (f.target_actual_operational.to_numpy(float) - current) / scale
        for start in FOLDS:
            tr = training_mask(f, start)
            va = month_mask(f, start)
            m = fit_cat(x[tr], y[tr], scale[tr], task)
            m.update(columns=names, cutoff=start)
            value = predict_cat(m, x[va], current[va], scale[va])
            models[(task, start)] = m
            if task == "direct":
                anchor[va] = value
            else:
                av.loc[va, "spatial_raw"] = value
                if start == FOLDS[0]:
                    av.loc[va, "spatial_full"] = value
                    av.loc[va, "spatial_current"] = value
                else:
                    cf = f.copy()
                    cf["available_pred"] = av.spatial_raw
                    cf["risk_probability"] = f.risk_q
                    cf["available_fold_start"] = pd.NaT
                    cf["available_training_latest_label"] = pd.NaT
                    for previous in FOLDS:
                        if previous >= start:
                            break
                        mask = month_mask(f, previous)
                        cf.loc[mask, "available_fold_start"] = pd.Timestamp(previous)
                        cf.loc[mask, "available_training_latest_label"] = f.loc[
                            training_mask(f, previous), "target_available_at"
                        ].max()
                    ct = (
                        tr
                        & f.Timestamp.ge(FOLDS[0]).to_numpy()
                        & np.isfinite(cf.available_pred)
                        & np.isfinite(cf.risk_q)
                    )
                    for route, riskrole in [("full", "cal"), ("current", "current")]:
                        center = fit_center(
                            cf.loc[ct].reset_index(drop=True), "A_innovation", riskrole, 0.1, start
                        )
                        centers[(route, start)] = center
                        av.loc[va, "spatial_" + route] = predict_center(center, cf.loc[va], 0.5)
            joblib.dump(m, out / (task + "_" + start + ".joblib"), compress=3)
            note(
                task,
                cutoff=start,
                train_rows=int(tr.sum()),
                latest_label=str(f.loc[tr, "target_available_at"].max()),
            )
        del x
    public_keys(av).to_csv(out / "available_oof.csv.gz", index=False)
    # Reconstruct all arrived-feedback states, beginning in September.
    online = np.full(len(f), np.nan)
    width = np.full((len(f), 2), np.nan)
    ds = qs = dp = qp = None
    jan_direct = jan_quality = None
    for phase, start, end in [
        ("early", "1997-09-01", "1997-11-01"),
        ("warm", "1997-11-01", "1998-01-01"),
        ("january", "1998-01-01", "1998-02-01"),
        ("test", "1998-02-01", "1998-05-12"),
    ]:
        hit = f.Timestamp.ge(start) & f.Timestamp.lt(end)
        ff = f.loc[hit].reset_index(drop=True)
        aa = av.loc[hit].reset_index(drop=True)
        facts = feedback(ff)
        r = direct_replay(
            ff,
            anchor[hit],
            query(ff, aa, anchor[hit], "observed"),
            facts,
            state=ds,
            pending_feedback=dp,
        )
        ds = r["state"]
        dp = facts
        online[hit] = r["predictions"]
        r = quality_replay(
            ff,
            aa[["spatial_full", "spatial_current"]].to_numpy(float),
            facts,
            state=qs,
            pending_feedback=qp,
        )
        qs = r["state"]
        qp = facts
        width[hit] = r["width"]
        if phase == "january":
            jan_direct = deepcopy(ds)
            jan_quality = deepcopy(qs)
        note("online_history", phase=phase, rows=len(ff))
    tools = {}
    columns = history_columns({"columns": schema["columns"]})
    farms = sorted(f.farm.unique())
    for role in ROLES:
        x, names = features(f, av, online, anchor, role, columns, farms)
        y = (f.target_actual_operational.to_numpy(float) - online) / scale
        for start in STARTS:
            tr = training_mask(f, start) & f.Timestamp.ge("1997-09-01").to_numpy()
            m = fit_tools(x[tr], y[tr], scale[tr])
            m.update(columns=names, history_columns=columns, farms=farms, cutoff=start, role=role)
            tools[(role, start)] = m
            joblib.dump(m, out / (role + "_" + start + ".joblib"), compress=3)
            note(
                "stage3_tools",
                role=role,
                cutoff=start,
                train_rows=int(tr.sum()),
                latest_label=str(f.loc[tr, "target_available_at"].max()),
            )
    hit = f.Timestamp.ge("1998-02-01")
    test = f.loc[hit].reset_index(drop=True)
    p = test[KEYS].copy()
    p["risk_q"] = test.risk_q.to_numpy()
    p["available_full"] = av.loc[hit, "spatial_full"].to_numpy()
    p["available_current"] = av.loc[hit, "spatial_current"].to_numpy()
    p["available_raw"] = av.loc[hit, "spatial_raw"].to_numpy()
    p["direct_static"] = anchor[hit]
    p["strong_online"] = online[hit]
    jan_agents = {}
    for role in ROLES:
        state = pending = None
        for phase, start, end in [
            ("warm", "1997-11-01", "1998-01-01"),
            ("january", "1998-01-01", "1998-02-01"),
            ("test", "1998-02-01", "1998-05-12"),
        ]:
            take = f.Timestamp.ge(start) & f.Timestamp.lt(end)
            ff = f.loc[take].reset_index(drop=True)
            aa = av.loc[take].reset_index(drop=True)
            plans, executed, agree = phase_plans(
                tools,
                ff,
                aa,
                online[take],
                anchor[take],
                role,
                width[take, 1 if role == "II_current" else 0],
            )
            facts = feedback(ff)
            r = actor_replay(
                ff,
                online[take],
                plans,
                executed,
                contexts(ff, role),
                versions(ff),
                facts,
                state=state,
                pending_feedback=pending,
            )
            state = r["state"]
            pending = facts
            if phase == "january":
                jan_agents[role] = deepcopy(state)
            if phase == "test":
                for j, policy in enumerate(POLICIES):
                    p[policy + "/" + role] = r["predictions"][:, j]
                    p[policy + "/" + role + "_action"] = r["actions"][:, j]
                joblib.dump(state, out / (role + "_end_state.joblib"), compress=3)
            note("stage3_agent", role=role, phase=phase)
    bundle = dict(
        risk_model=bridges["final"],
        available_model=models[("available", "1998-02-01")],
        centers={role: centers[(role, "1998-02-01")] for role in ("full", "current")},
        direct_model=models[("direct", "1998-02-01")],
        observed=schema["observed"],
        available_observed=schema["columns"]["aggregate"],
        spatial_columns=list(sp),
        final_training_cutoff="1998-02-01",
        target="Actual power one hour after issue",
        version="F20_strong_online_increment",
        policies=POLICIES,
        tools={role: tools[(role, "1998-02-01")] for role in ROLES},
        states=dict(direct=jan_direct, quality=jan_quality, agents=jan_agents),
    )
    joblib.dump(bundle, out / "models_h4.joblib", compress=3)
    joblib.dump(centers, out / "stage2_centers.joblib", compress=3)
    # No expected outputs/models are read before this point.
    public_keys(p).to_csv(out / "predictions.csv.gz", index=False)
    rows = write_metrics(test, p, out)
    note(
        "training_finished",
        used_archived_models=False,
        used_archived_oof=False,
        used_archived_states=False,
    )
    report = compare(out, test, p, ROLES)
    write_json(
        out / "complete.json",
        dict(
            status="complete",
            mode="fresh training from public 15-minute SCADA observations",
            used_archived_models=False,
            used_archived_oof=False,
            used_archived_states=False,
            issued_rows=len(test),
            consistent_rows=int((eligible(test) & np.isfinite(test.target_potential)).sum()),
            elapsed_seconds=time.monotonic() - started,
            reference_verification=report,
            core_metrics=[
                r for r in rows if r["quality"] == "consistent" and r["group"] == "pooled"
            ],
        ),
    )
    print("FRESH TRAINING COMPLETE:", out, flush=True)
