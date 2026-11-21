"""Reconstruct features from anonymous observations, apply models, and score predictions."""

import joblib
import json
import numpy as np
import pandas as pd
import time
from .sequences import CORE, eligible, windows, write_json
from .roles import KEYS, POLICIES, ROLES, feedback, static_frame
from .available_features import build_available_features
from .risk_features import BASE_COLUMNS
from .grid import assign_phases, attach_targets
from .validation import align_reference, assign_shared, compare_series, validate_predictions
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


EPOCH = pd.Timestamp("1997-07-01")


STEP = pd.Timedelta(minutes=15)


def config():
    return json.loads(Path(__file__).with_name("protocol.json").read_text(encoding="utf-8"))


def load_schema():
    return json.loads(Path(__file__).with_name("features.json").read_text(encoding="utf-8"))


def new_output(path):
    result = Path(path).resolve()
    for reserved in ["src", "data", "reference", "models", "paper", "figures", "tables"]:
        if result.is_relative_to(ROOT / reserved):
            raise ValueError(f"Choose a new output directory outside {reserved}/.")
    result.mkdir(parents=True, exist_ok=False)
    return result


def read_observations():
    source = ROOT / "data" / "observations_15min.csv.gz"
    cols = ["slot" if name == "Timestamp" else name for name in BASE_COLUMNS]
    observations = pd.read_csv(source, usecols=cols, float_precision="round_trip")
    observations["Timestamp"] = EPOCH + observations.pop("slot") * STEP
    if set(observations.farm.unique()) != {"F1", "F2", "F3", "F4", "F5"}:
        raise ValueError("Expected the five public farm identifiers.")
    return observations[BASE_COLUMNS]


def read_reference():
    reference = pd.read_csv(ROOT / "reference/crhf_replay.csv.gz", float_precision="round_trip")
    for slot, col in [("slot", "Timestamp"), ("target_slot", "target_timestamp")]:
        reference[col] = EPOCH + reference.pop(slot) * STEP
    return reference


def public_keys(frame):
    result = frame.copy()
    for name, slot in [("Timestamp", "slot"), ("target_timestamp", "target_slot")]:
        if name in result:
            result[name] = (pd.to_datetime(result[name]) - EPOCH) // STEP
            result.rename(columns={name: slot}, inplace=True)
    return result


def reconstruct(sequence=True):
    """Build causal inputs on the full grid, then select valid issue rows in fit order."""
    cfg = config()
    schema = load_schema()
    base = read_observations()
    grid, risk_spec, power_spec = build_available_features(base, cfg)
    # Invalid grid slots remain present; no row compression for sequence windows.
    grid.loc[~grid.input_valid.astype(bool), CORE] = np.nan
    allf = assign_phases(attach_targets(grid, 4, cfg), cfg)
    parts = []
    for start, end in [
        ("1997-08-01", "1997-12-01"),
        ("1997-12-01", "1998-01-01"),
        ("1998-01-01", "1998-02-01"),
        ("1998-02-01", "1998-05-12"),
    ]:
        f = allf.loc[allf.input_valid & allf.Timestamp.ge(start) & allf.Timestamp.lt(end)].copy()
        if start == "1997-08-01":
            f["evaluation_eligible"] = f.sample_valid
            f["phase"] = "train"
        parts.append(f)
    f = pd.concat(parts, ignore_index=True)
    f["target_available_inconsistent"] = f.target_actual_operational.gt(
        f.target_potential + 0.03 * f.capacity_proxy
    )
    assert not f[KEYS].duplicated().any()
    assert f.target_timestamp.sub(f.Timestamp).eq(pd.Timedelta(hours=1)).all()
    assert f.observation_available_at.le(f.Timestamp).all()
    assert len(f) == 120831, (
        "Data/version changed: this entry reproduces the frozen five-farm experiment"
    )
    assert all(c in f for c in schema["observed"])
    seq = windows(grid, f) if sequence else None
    source = base[["farm", "Timestamp", "potential", "capacity_proxy", "input_valid"]].copy()
    return dict(
        frame=f,
        sequence=seq,
        grid=grid,
        all_frame=allf,
        source=source,
        config=cfg,
        schema=schema,
        risk_spec=risk_spec,
        power_spec=power_spec,
    )


def write_metrics(frame, predictions, out):
    from sklearn.metrics import average_precision_score, brier_score_loss

    validate_predictions(frame, predictions)
    f = frame
    rows = []
    for quality in ("consistent", "all_valid"):
        valid = eligible(f, consistent=quality == "consistent") & np.isfinite(f.target_potential)
        groups = {"pooled": np.ones(len(f), bool)}
        groups.update({"farm=" + v: f.farm.eq(v).to_numpy() for v in sorted(f.farm.unique())})
        groups.update(
            {
                "month=" + v: f.Timestamp.dt.strftime("%Y-%m").eq(v).to_numpy()
                for v in sorted(f.Timestamp.dt.strftime("%Y-%m").unique())
            }
        )
        for group, mask in groups.items():
            take = valid & mask
            for col in predictions:
                if col in KEYS or col.endswith("_action"):
                    continue
                if col == "risk_q":
                    y = f.loc[take, "target_curtailed"]
                    p = predictions.loc[take, col]
                    rows.append(
                        dict(
                            stage=1,
                            model=col,
                            quality=quality,
                            group=group,
                            n=int(take.sum()),
                            AP=average_precision_score(y, p),
                            Brier=brier_score_loss(y, p),
                        )
                    )
                else:
                    stage = 2 if col.startswith("available_") else 3
                    target = "target_potential" if stage == 2 else "target_actual_operational"
                    err = predictions.loc[take, col].to_numpy(float) - f.loc[
                        take, target
                    ].to_numpy(float)
                    rows.append(
                        dict(
                            stage=stage,
                            model=col,
                            quality=quality,
                            group=group,
                            n=len(err),
                            MAE_MW=float(np.mean(abs(err))),
                            RMSE_MW=float(np.sqrt(np.mean(err**2))),
                            bias_MW=float(np.mean(err)),
                        )
                    )
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    return rows


def compare(out, frame, predictions, roles, atol=3e-5):
    # Deliberately called AFTER predictions.csv.gz is on disk.
    validate_predictions(frame, predictions)
    if not roles or len(set(roles)) != len(roles) or not set(roles).issubset(ROLES):
        raise ValueError("Expected distinct registered ablation roles")
    records = []
    reference = read_reference()
    reference = align_reference(frame, reference)
    expected = [
        "risk_q",
        "available_full",
        "available_current",
        "available_raw",
        "direct_static",
        "strong_online",
    ]
    for role in roles:
        for policy in POLICIES:
            name = policy + "/" + role
            expected.extend([name, name + "_action"])
    if set(predictions.columns) != set(KEYS + expected):
        raise ValueError("Missing or unexpected prediction series for requested roles")
    for name in expected:
        record = compare_series(
            predictions[name],
            reference[name],
            name,
            atol=min(atol, 1e-10) if name == "risk_q" else atol,
            action=name.endswith("_action"),
        )
        if record.get("unit") == "MW":
            record["max_abs_error_MW"] = record["max_abs_error"]
        records.append(record)
    report = dict(
        status="verified",
        rows=len(frame),
        reference_read_after_forecast=True,
        tolerance_MW=atol,
        series=records,
    )
    write_json(out / "reference_comparison.json", report)
    return report


def add_role_predictions(frame, role, replay_result):
    for name, value in {
        "risk_q": replay_result["risk"],
        "available_full": replay_result["available"].spatial_full,
        "available_current": replay_result["available"].spatial_current,
        "available_raw": replay_result["available_raw"],
        "direct_static": replay_result["static"],
        "strong_online": replay_result["online"]["predictions"],
    }.items():
        assign_shared(frame, name, value)
    for policy_index, policy in enumerate(POLICIES):
        frame[f"{policy}/{role}"] = replay_result["predictions"][:, policy_index]
        frame[f"{policy}/{role}_action"] = replay_result["actions"][:, policy_index]


def replay(out, roles=ROLES, until=None, bundle_path=None):
    from .inference import saved_predict
    import torch

    torch.set_num_threads(2)
    d = reconstruct()
    allf = d["frame"]
    hit = allf.Timestamp.ge("1998-02-01")
    if until:
        hit &= allf.Timestamp.lt(until)
    f = allf.loc[hit].reset_index(drop=True)
    seq = d["sequence"][hit]
    pending = feedback(allf.loc[allf.Timestamp.ge("1998-01-01") & allf.Timestamp.lt("1998-02-01")])
    b = joblib.load(bundle_path or ROOT / "models/crhf.joblib")
    p = f[KEYS].copy()
    for role in roles:
        print("Computing saved three-stage inference:", role, len(f), flush=True)
        r = saved_predict(
            b, static_frame(f), seq, d["source"], feedback(f), role=role, pending_feedback=pending
        )
        add_role_predictions(p, role, r)
        joblib.dump(r["state"], out / (role + "_end_state.joblib"), compress=3)
        np.savez_compressed(
            out / (role + "_audit.npz"),
            actions=r["actions"],
            scores=r["scores"],
            counts=r["counts"],
            days=r["days"],
        )
    public_keys(p).to_csv(out / "predictions.csv.gz", index=False)
    rows = write_metrics(f, p, out)
    report = compare(out, f, p, roles)
    write_json(
        out / "complete.json",
        dict(
            status="complete",
            mode="saved-model replay",
            roles=list(roles),
            issued_rows=len(f),
            consistent_rows=int((eligible(f) & np.isfinite(f.target_potential)).sum()),
            reference_verification=report,
            core_metrics=[
                r for r in rows if r["quality"] == "consistent" and r["group"] == "pooled"
            ],
        ),
    )
    print("COMPLETE:", out, flush=True)
