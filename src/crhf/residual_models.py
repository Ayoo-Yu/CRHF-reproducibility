"""Train the robust linear correction and three-seed neural correction on forward residuals."""

import numpy as np
import torch
from .roles import CAP, ROLES, SEEDS, train_transform, transform
from .available_residual import BASE_COLUMNS
from torch import nn


def history_columns(d):
    return list(dict.fromkeys([*d["columns"]["own"], *BASE_COLUMNS]))


def role_values(f, available, role, override=None):
    if role not in ROLES:
        raise ValueError(role)
    q = f.teacher_probability.to_numpy(float) if role == "observed" else f.risk_q.to_numpy(float)
    if role in ("no_available", "observed"):
        a = f.potential.to_numpy(float)
    elif override is not None:
        a = np.asarray(override, float)
    else:
        a = np.asarray(
            available["spatial_current" if role == "II_current" else "spatial_full"], float
        )
    return a, q


def features(f, available, online, static, role, columns, farms, override=None):
    a, q = role_values(f, available, role, override)
    s = f.capacity_proxy.to_numpy(float)
    a0 = f.potential.to_numpy(float)
    p0 = f.actual_operational.to_numpy(float)
    delta = (a - a0) / s
    q0 = f.teacher_probability.to_numpy(float)
    onehot = np.column_stack([f.farm.eq(name).to_numpy(float) for name in farms])
    interface = np.c_[
        online / s,
        (static - p0) / s,
        (online - static) / s,
        q,
        q - q0,
        a / s,
        delta,
        (a - online) / s,
        (q - q0) * delta,
        (1 - q) * delta,
    ]
    names = (
        columns
        + ["farm_" + v for v in farms]
        + [
            "online_level",
            "static_change",
            "online_correction",
            "future_q",
            "risk_change",
            "future_A_level",
            "future_A_change",
            "future_A_minus_online",
            "risk_A_interaction",
            "clear_A_interaction",
        ]
    )
    return np.c_[f[columns].to_numpy(float), onehot, interface].astype(np.float32), names


def contexts(f, role):
    q = f.teacher_probability.to_numpy(float) if role == "observed" else f.risk_q.to_numpy(float)
    return 2 * f.teacher_probability.ge(0.5).to_numpy(int) + (q >= 0.5).astype(int)


class CorrectionNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_tools(x, y, scale):
    t = train_transform(x)
    raw = transform(x, t)
    z = np.c_[np.ones(len(raw)), raw].astype(float)
    weight = scale / scale.sum()
    ridge = np.eye(z.shape[1]) * 0.1
    ridge[0, 0] = 0.0
    beta = np.zeros(z.shape[1])
    for _ in range(18):
        w = weight / np.sqrt((y - z @ beta) ** 2 + 0.005**2)
        beta = np.linalg.solve(z.T @ (w[:, None] * z) + ridge, z.T @ (w * y))
    torch.set_num_threads(2)
    tx = torch.from_numpy(raw)
    ty = torch.from_numpy(y.astype(np.float32))
    tw = torch.from_numpy((scale / scale.mean()).astype(np.float32))
    nets = []
    losses = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = CorrectionNet(raw.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=0.0003, weight_decay=0.0001)
        for epoch in range(40):
            order = rng.permutation(len(raw))
            total = 0.0
            for start in range(0, len(raw), 512):
                ix = order[start : start + 512]
                opt.zero_grad(set_to_none=True)
                loss = (torch.abs(net(tx[ix]) - ty[ix]) * tw[ix]).mean()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(ix)
            losses.append(
                dict(seed=seed, epoch=epoch + 1, weighted_normalized_MAE=total / len(raw))
            )
        nets.append({k: v.detach().cpu().numpy() for k, v in net.state_dict().items()})
    return dict(transform=t, beta=beta, nets=nets, input_dim=raw.shape[1], losses=losses)


def predict_deltas(model, x):
    raw = transform(x, model["transform"])
    linear = np.c_[np.ones(len(raw)), raw] @ model["beta"]
    pred = []
    for weights in model["nets"]:
        net = CorrectionNet(model["input_dim"])
        net.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
        net.eval()
        with torch.no_grad():
            pred.append(
                np.concatenate(
                    [
                        net(torch.from_numpy(raw[i : i + 4096])).numpy().astype(float)
                        for i in range(0, len(raw), 4096)
                    ]
                )
            )
    return np.clip(np.c_[linear, np.mean(pred, axis=0)], -CAP, CAP)
