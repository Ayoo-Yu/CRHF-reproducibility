"""CPU training of direct power-increment predictors."""
from copy import deepcopy

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .models import ADAMW_MODELS, SEEDS, ForecastNet, tree_model


def predict_network(model, batch):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(batch.y), 512):
            sl = slice(start, start + 512)
            output = model(torch.from_numpy(batch.seq[sl]), torch.from_numpy(batch.tab[sl]))
            values.append(output.numpy())
    return np.concatenate(values)


def fit_network(name, cfg, train, seed, epochs, validation=None):
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    net = ForecastNet(name, train.tab.shape[1], cfg)
    optimizer = torch.optim.AdamW(net.parameters(), lr=(.001, .0003)[cfg], weight_decay=1e-4)
    tensors = list(map(torch.from_numpy, (
        train.seq, train.tab, train.y, train.scale / train.scale.mean())))
    seq, features, target, weight = tensors
    best = np.inf
    best_epoch = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        net.train()
        order = generator.permutation(len(target))
        for start in range(0, len(target), 512):
            rows = order[start:start + 512]
            optimizer.zero_grad(set_to_none=True)
            estimate = net(seq[rows], features[rows])
            loss = ((estimate - target[rows]).abs() * weight[rows]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss for {name}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optimizer.step()
        if validation is None:
            continue
        increment = predict_network(net, validation)
        prediction = np.maximum(validation.base + validation.scale * increment, 0)
        observed = validation.base + validation.scale * validation.y
        mae = float(np.mean(np.abs(prediction - observed)))
        if mae < best - 1e-6:
            best, best_epoch, best_state = mae, epoch, deepcopy(net.state_dict())
        elif epoch - best_epoch >= 6:
            break
    if validation is not None:
        net.load_state_dict(best_state)
    return net, (best_epoch if validation is not None else epochs)


def fit_predict(name, cfg, train, evaluate, seed=SEEDS[0], epochs=None, select=False):
    best_epoch = None
    if name in ADAMW_MODELS:
        net, best_epoch = fit_network(name, cfg, train, seed, epochs or 40,
                                      validation=evaluate if select else None)
        increments = predict_network(net, evaluate)
    else:
        model = tree_model(name, cfg, seed)
        with threadpool_limits(6):
            model.fit(train.flat(), train.y, sample_weight=train.scale / train.scale.mean())
            increments = model.predict(evaluate.flat())
    pred = np.maximum(evaluate.base + evaluate.scale * increments, 0)
    if not np.isfinite(pred).all():
        raise FloatingPointError(f"Non-finite prediction for {name}")
    return pred, best_epoch


def select_configuration(name, train, validation):
    observed = validation.base + validation.scale * validation.y
    candidates = []
    for cfg in (0, 1):
        pred, epoch = fit_predict(name, cfg, train, validation, select=True)
        candidates.append((float(np.abs(observed - pred).mean()), cfg, epoch))
    _, cfg, epoch = min(candidates)
    return cfg, epoch
