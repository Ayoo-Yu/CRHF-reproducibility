"""Replay a trained CRHF model on histories ending at the current forecast origin."""

import numpy as np
import pandas as pd
from .local_feedback import replay as online_replay
from .spatial import spatial_features
from .backbones import predict_cat
from .interfaces import query
from .correction_selection import replay as actor_replay
from .roles import ROLES, static_frame, versions
from .residual_models import contexts
from .proposals import proposals
from .available_feedback import replay as quality_replay
from .risk_bridge import predict_risk
from .available_residual import predict as center_predict


def saved_predict(
    bundle,
    raw_frame,
    sequence,
    spatial_source,
    facts,
    *,
    role="full",
    state=None,
    pending_feedback=None,
    bias=0.0,
    asof=None,
):
    if role not in ROLES:
        raise ValueError(role)
    if bias not in (0.0, -0.05, 0.05) or (bias and role != "full"):
        raise ValueError("Unregistered bias")
    f = static_frame(raw_frame).copy().reset_index(drop=True)
    if len(f) == 0 or f.Timestamp.min() < pd.Timestamp(bundle["final_training_cutoff"]):
        raise ValueError("Final model before training cutoff")
    if not f.observation_available_at.le(f.Timestamp).all():
        raise ValueError("Future observation")
    f["target_timestamp"] = f.Timestamp + pd.Timedelta(hours=1)
    f["risk_raw_f19"], f["risk_q"] = predict_risk(bundle["risk_model"], f)
    sp = spatial_features(spatial_source, f).astype(np.float32)
    if list(sp) != bundle["spatial_columns"]:
        raise ValueError("Changed spatial station schema")
    model = bundle["available_model"]
    x = pd.concat([f[bundle["available_observed"]], sp], axis=1)[model["columns"]].to_numpy(
        np.float32
    )
    scale = f.capacity_proxy.to_numpy(float)
    raw_available = predict_cat(model, x, f.potential.to_numpy(float), scale)
    f["available_pred"] = raw_available
    f["risk_probability"] = f.risk_q
    av = pd.DataFrame(
        {
            "spatial_" + name: center_predict(bundle["centers"][name], f, 0.5)
            for name in ("full", "current")
        }
    )
    seq = np.asarray(sequence, np.float32)
    if seq.shape != (len(f), 96, 6):
        raise ValueError("Expected 96 by 6 causal histories ending at each issue")
    dx = np.c_[
        seq.reshape(len(f), -1),
        f[bundle["observed"]].to_numpy(np.float32),
        sp.to_numpy(np.float32),
    ]
    anchor = predict_cat(bundle["direct_model"], dx, f.actual_operational.to_numpy(float), scale)
    initial = bundle["states"] if state is None else state
    actor_state = initial["agents"][role] if state is None else initial["agent"]
    q = query(f, av, anchor, "observed")
    online = online_replay(
        f, anchor, q, facts, state=initial["direct"], pending_feedback=pending_feedback, asof=asof
    )
    # Only the stress experiment biases the uncertainty history; nominal Stage II stays visible.
    qa = np.maximum(
        av[["spatial_full", "spatial_current"]].to_numpy(float) + bias * scale[:, None], 0
    )
    quality = quality_replay(
        f, qa, facts, state=initial["quality"], pending_feedback=pending_feedback, asof=asof
    )
    width = quality["width"][:, 1 if role == "II_current" else 0]
    prop = proposals(
        bundle["tools"][role], f, av, online["predictions"], anchor, role, width, bias
    )
    actor = actor_replay(
        f,
        online["predictions"],
        prop["plans"],
        prop["executed"],
        contexts(f, role),
        versions(f),
        facts,
        state=actor_state,
        pending_feedback=pending_feedback,
        asof=asof,
    )
    actor_state = actor.pop("state")
    return dict(
        **actor,
        state=dict(direct=online.pop("state"), quality=quality.pop("state"), agent=actor_state),
        available=av,
        available_raw=raw_available,
        risk=f.risk_q.to_numpy(float),
        static=anchor,
        online=online,
        quality=quality,
        tools=prop["plans"],
        executed=prop["executed"],
        sign_agreement=prop["sign_agreement"],
    )
