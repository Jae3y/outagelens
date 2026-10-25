"""Model 3: tree-structured message-passing Graph Neural Network (pure NumPy).

A plain GCN needs as many layers as the feeder is deep before a leaf meter can
influence the trunk. Distribution feeders are rooted trees, so each layer here
aggregates over three topology-aware neighbourhoods at once:

    H' = ReLU( H W_self + S H W_subtree + P H W_path + A H W_nbr + b )

* S: mean over the node's downstream subtree (what a fault at this node would cut off)
* P: mean over the path back to the substation (what must still be energized)
* A: symmetric-normalised adjacency (local smoothing)

Two layers give every node a feeder-wide receptive field. Each section
e = (u, v) is scored from its endpoint embeddings, s_e = h_v.a + h_u.b + c, and
a softmax over sections gives P(e | D). Forward and backward passes are written
by hand (verified by a finite-difference gradient check in tests), trained with
Adam, so the model needs no compiled dependency and trains in seconds on a CPU.

If PyTorch Geometric is installed the same architecture can be ported 1:1; the
NumPy version is kept as the default because it is guaranteed to run anywhere.
"""
from __future__ import annotations

import numpy as np

from .ensemble import combine
from .feeder_topology import Feeder
from .sparsity_simulator import MISSING, Scenario, fit_temperature, random_config, simulate

FEATURE_NAMES = [
    "Outage ping  I[S=0]",
    "Healthy ping  I[S=1]",
    "Meter missing  I[S=None]",
    "Post-event voltage V (0 if unmonitored)",
    "Voltage shift ΔV (x20)",
    "Voltage sensor present",
    "Nominal load P/Pmax",
    "Depth from substation (structural)",
    "Downstream load share (structural)",
    "Feeder-head shed fraction (SCADA, global)",
]


class TreeGNN:
    def __init__(self, feeder: Feeder, hidden: int = 32, seed: int = 0):
        self.f = feeder
        self.hidden = hidden
        n = feeder.n_bus
        idx = feeder.bus_index

        sub = feeder.downstream.astype(float)  # rows = sections, cols = buses in subtree of child
        S = np.zeros((n, n))
        S[feeder.edge_child_idx] = sub
        S[idx[feeder.source]] = 1.0  # the substation's subtree is the whole feeder
        self.S = S / S.sum(axis=1, keepdims=True)

        P = np.zeros((n, n))
        for b in feeder.buses:
            for a in feeder.ancestors[b] + [b]:
                P[idx[b], idx[a]] = 1.0
        self.P = P / P.sum(axis=1, keepdims=True)

        A = np.eye(n)
        A[feeder.edge_child_idx, feeder.edge_parent_idx] = 1.0
        A[feeder.edge_parent_idx, feeder.edge_child_idx] = 1.0
        d = 1.0 / np.sqrt(A.sum(axis=1))
        self.A = A * d[:, None] * d[None, :]

        self.Cm = np.zeros((feeder.n_edge, n))
        self.Cm[np.arange(feeder.n_edge), feeder.edge_child_idx] = 1.0
        self.Pm = np.zeros((feeder.n_edge, n))
        self.Pm[np.arange(feeder.n_edge), feeder.edge_parent_idx] = 1.0

        p = feeder.p_kw
        depth = np.array([feeder.depth[b] for b in feeder.buses], dtype=float)
        sub_load = np.zeros(n)
        sub_load[feeder.edge_child_idx] = feeder.shed_kw
        sub_load[idx[feeder.source]] = p.sum()
        self.static = np.stack([p / max(p.max(), 1e-9), depth / depth.max(), sub_load / p.sum()], axis=1)

        self.n_feat = len(FEATURE_NAMES)
        rng = np.random.default_rng(seed)
        self.params = {
            "W0": rng.normal(0, np.sqrt(2.0 / (4 * self.n_feat)), (4, self.n_feat, hidden)),
            "b0": np.zeros(hidden),
            "W1": rng.normal(0, np.sqrt(2.0 / (4 * hidden)), (4, hidden, hidden)),
            "b1": np.zeros(hidden),
            "a": rng.normal(0, np.sqrt(1.0 / hidden), hidden),
            "bv": rng.normal(0, np.sqrt(1.0 / hidden), hidden),
            "c": np.zeros(1),
        }
        self.history: list[dict] = []
        self.temperature = 1.0
        self.trained = False

    # -------------------------------------------------------------- features
    def features(self, sc: Scenario) -> np.ndarray:
        n = self.f.n_bus
        x = np.zeros((n, self.n_feat))
        x[:, 0] = sc.meter == 0
        x[:, 1] = sc.meter == 1
        x[:, 2] = (sc.meter == MISSING) & sc.has_meter
        x[:, 3] = np.nan_to_num(sc.v_post_obs, nan=0.0)
        x[:, 4] = np.nan_to_num(sc.dv_obs, nan=0.0) * 20.0
        x[:, 5] = sc.sensor
        x[:, 6] = self.static[:, 0]
        x[:, 7] = self.static[:, 1]
        x[:, 8] = self.static[:, 2]
        x[:, 9] = sc.shed_kw_obs / max(sc.head_kw_pre_obs, 1e-9)
        return x

    # ------------------------------------------------------- forward/backward
    @staticmethod
    def _prop(M: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Apply a node operator M (R x N) to a batch H (B, N, F) as one GEMM."""
        B, N, F = H.shape
        out = M @ H.transpose(1, 0, 2).reshape(N, B * F)
        return out.reshape(M.shape[0], B, F).transpose(1, 0, 2)

    def _forward(self, X: np.ndarray):
        cache = []
        H = X
        for layer in (0, 1):
            W, b = self.params[f"W{layer}"], self.params[f"b{layer}"]
            aggs = (H, self._prop(self.S, H), self._prop(self.P, H), self._prop(self.A, H))
            cat = np.concatenate(aggs, axis=-1)  # (B, N, 4F): one GEMM for all four message types
            Z = (cat.reshape(-1, cat.shape[-1]) @ W.reshape(-1, W.shape[-1])).reshape(*H.shape[:2], -1) + b
            cache.append((aggs, Z))
            H = np.maximum(Z, 0.0)
        hc = self._prop(self.Cm, H)  # (B, E, hidden)
        hp = self._prop(self.Pm, H)
        s = hc @ self.params["a"] + hp @ self.params["bv"] + self.params["c"][0]
        return s, (cache, hc, hp)

    @staticmethod
    def _softmax(s: np.ndarray) -> np.ndarray:
        z = np.exp(s - s.max(axis=-1, keepdims=True))
        return z / z.sum(axis=-1, keepdims=True)

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray):
        B = X.shape[0]
        s, (cache, hc, hp) = self._forward(X)
        prob = self._softmax(s)
        loss = float(-np.log(prob[np.arange(B), y] + 1e-12).mean())
        ds = prob.copy()
        ds[np.arange(B), y] -= 1.0
        ds /= B

        g = {"a": np.einsum("be,beh->h", ds, hc), "bv": np.einsum("be,beh->h", ds, hp),
             "c": np.array([ds.sum()])}
        dH = (self._prop(self.Cm.T, ds[..., None] * self.params["a"])
              + self._prop(self.Pm.T, ds[..., None] * self.params["bv"]))
        for layer in (1, 0):
            aggs, Z = cache[layer]
            W = self.params[f"W{layer}"]
            dZ = dH * (Z > 0)
            dZf = dZ.reshape(-1, dZ.shape[-1])
            g[f"W{layer}"] = np.stack([agg.reshape(-1, agg.shape[-1]).T @ dZf for agg in aggs])
            g[f"b{layer}"] = dZ.sum(axis=(0, 1))
            if layer > 0:
                dH = (dZ @ W[0].T + self._prop(self.S.T, dZ @ W[1].T) + self._prop(self.P.T, dZ @ W[2].T)
                      + self._prop(self.A.T, dZ @ W[3].T))
        return loss, g, prob

    def predict(self, sc: Scenario) -> np.ndarray:
        s, _ = self._forward(self.features(sc)[None])
        return self._softmax(s / self.temperature)[0]

    # ------------------------------------------------------------- training
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 60, batch: int = 64, lr: float = 0.01,
            weight_decay: float = 1e-4, val_frac: float = 0.15, seed: int = 0, progress=None) -> None:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(y))
        n_val = int(len(y) * val_frac)
        val, tr = order[:n_val], order[n_val:]
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v2 = {k: np.zeros_like(v) for k, v in self.params.items()}
        b1, b2, t = 0.9, 0.999, 0
        self.history = []
        for ep in range(epochs):
            rng.shuffle(tr)
            losses = []
            for i in range(0, len(tr), batch):
                bi = tr[i:i + batch]
                loss, g, _ = self.loss_and_grads(X[bi], y[bi])
                losses.append(loss)
                t += 1
                step = lr * (0.5 * (1 + np.cos(np.pi * ep / epochs)))  # cosine decay
                for k in self.params:
                    gk = g[k] + weight_decay * self.params[k]
                    m[k] = b1 * m[k] + (1 - b1) * gk
                    v2[k] = b2 * v2[k] + (1 - b2) * gk**2
                    mh, vh = m[k] / (1 - b1**t), v2[k] / (1 - b2**t)
                    self.params[k] -= step * mh / (np.sqrt(vh) + 1e-8)
            s_val, _ = self._forward(X[val])
            p_val = self._softmax(s_val)
            val_acc = float((p_val.argmax(axis=1) == y[val]).mean()) if n_val else float("nan")
            top3 = float(np.mean([y[val][j] in np.argsort(-p_val[j])[:3] for j in range(n_val)])) if n_val else 0
            self.history.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)),
                                 "val_top1": val_acc, "val_top3": top3})
            if progress is not None:
                progress((ep + 1) / epochs, f"Epoch {ep + 1}/{epochs}  loss {np.mean(losses):.3f}  "
                                            f"val top-1 {val_acc:.0%}  top-3 {top3:.0%}")
        if n_val:
            self.temperature = fit_temperature(self._forward(X[val])[0], y[val])
        self.trained = True


def build_training_set(feeder: Feeder, engine, n: int = 600, seed: int = 7, progress=None):
    """Balanced corpus: every section faulted equally often under random observability."""
    rng = np.random.default_rng(seed)
    gnn_tmp = TreeGNN(feeder)
    X = np.zeros((n, feeder.n_bus, gnn_tmp.n_feat))
    y = np.zeros(n, dtype=int)
    faults = np.resize(rng.permutation(feeder.n_edge), n)
    for i in range(n):
        sc = simulate(feeder, engine, int(faults[i]), random_config(rng), seed=int(rng.integers(1 << 31)))
        X[i] = gnn_tmp.features(sc)
        y[i] = faults[i]
        if progress is not None and (i % 25 == 0 or i == n - 1):
            progress((i + 1) / n, f"Simulating fault scenario {i + 1}/{n}")
    return X, y


MODEL_VERSION = "treegnn-v2"


def train_gnn(feeder: Feeder, engine, n_scenarios: int = 600, epochs: int = 40, hidden: int = 24,
              batch: int = 128, lr: float = 0.02, seed: int = 7, progress=None) -> TreeGNN:
    def sim_progress(frac, msg):
        if progress:
            progress(0.2 * frac, msg)

    def fit_progress(frac, msg):
        if progress:
            progress(0.2 + 0.8 * frac, msg)

    X, y = build_training_set(feeder, engine, n_scenarios, seed, sim_progress)
    model = TreeGNN(feeder, hidden=hidden, seed=seed)
    model.fit(X, y, epochs=epochs, batch=batch, lr=lr, seed=seed, progress=fit_progress)
    return model


def save_model(model: TreeGNN, path) -> None:
    hist = np.array([[h["epoch"], h["train_loss"], h["val_top1"], h["val_top3"]] for h in model.history])
    np.savez(path, version=MODEL_VERSION, feeder=model.f.key, hidden=model.hidden, history=hist,
             temperature=model.temperature, **{f"p_{k}": v for k, v in model.params.items()})


def load_model(feeder: Feeder, path) -> TreeGNN | None:
    try:
        data = np.load(path, allow_pickle=False)
        if str(data["version"]) != MODEL_VERSION or str(data["feeder"]) != feeder.key:
            return None
        model = TreeGNN(feeder, hidden=int(data["hidden"]))
        for k in model.params:
            model.params[k] = data[f"p_{k}"]
        model.history = [{"epoch": int(r[0]), "train_loss": r[1], "val_top1": r[2], "val_top3": r[3]}
                         for r in data["history"]]
        model.temperature = float(data["temperature"])
        model.trained = True
        return model
    except (OSError, KeyError, ValueError):
        return None


def ensemble(p_bayes: np.ndarray, p_gnn: np.ndarray) -> np.ndarray:
    """OutageLens ensemble: margin-gated pooling of Bayes and GNN (see ``core.ensemble``)."""
    return combine(p_bayes, p_gnn)
