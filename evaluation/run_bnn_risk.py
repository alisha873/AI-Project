"""AutoShield Phase 16: Bayes-by-Backprop neural network.

Labels are known scenario edge assessments, not post-remediation residual
risk labels. Results must be interpreted as attack-risk proxies.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, mean_absolute_error, roc_auc_score,
)

from evidence.schema import AttackGraph
from evaluation.gnn_dataset import (
    ScenarioGraphRecord, ScenarioGroundTruth, ScenarioSplit,
    prepare_edge_examples,
)
from graph.gnn import encode_graph, EDGE_TYPES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evaluation" / "scenarios"
OUT = ROOT / "evaluation" / "results" / "bnn_risk_results.json"
MODEL_OUT = ROOT / "evaluation" / "results" / "models" / "bayesian_risk_model.pt"

SEED = 7
EPOCHS = 500
SAMPLES = 500
PATIENCE = 60
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -------------------- Load leakage-aware scenario data --------------------

assignments = [
    ScenarioSplit.model_validate(x)
    for x in json.loads((DATA / "gnn_splits.json").read_text())
]

graphs, truths = [], []
for assignment in assignments:
    sid = assignment.scenario_id
    graph_raw = json.loads((DATA / f"{sid}_graph.json").read_text())
    truth_raw = json.loads((DATA / f"{sid}_ground_truth.json").read_text())
    graphs.append(ScenarioGraphRecord.model_validate({
        "scenario_id": sid, "graph": graph_raw
    }))
    truths.append(ScenarioGroundTruth.model_validate(truth_raw))

examples = prepare_edge_examples(graphs, truths, assignments)
graph_by_id = {r.scenario_id: r.graph for r in graphs}
split_by_id = {a.scenario_id: a.split for a in assignments}


# -------------------- Build real graph-derived edge features --------------------

def features_for_edge(graph, source_id, target_id, edge_type):
    """Use encoded node evidence, topology, relation, and edge basis."""
    tensors = encode_graph(graph)
    node_index = {node_id: i for i, node_id in enumerate(tensors.node_ids)}
    if source_id not in node_index or target_id not in node_index:
        raise ValueError(f"Missing edge endpoint: {source_id} -> {target_id}")

    src_i, dst_i = node_index[source_id], node_index[target_id]
    src = tensors.node_features[src_i].numpy()
    dst = tensors.node_features[dst_i].numpy()

    edge = next(
        e for e in graph.edges
        if e.source == source_id and e.target == target_id
        and e.edge_type == edge_type
    )

    relation = np.zeros(len(EDGE_TYPES), dtype=np.float32)
    relation[EDGE_TYPES.index(edge_type)] = 1.0
    basis = np.array([
        float(edge.basis == "configured"),
        float(edge.basis == "observed"),
    ], dtype=np.float32)

    # Normalized local topology, independent of ground-truth labels.
    n = max(len(graph.nodes), 1)
    src_out = sum(e.source == source_id for e in graph.edges) / n
    src_in = sum(e.target == source_id for e in graph.edges) / n
    dst_out = sum(e.source == target_id for e in graph.edges) / n
    dst_in = sum(e.target == target_id for e in graph.edges) / n
    topology = np.asarray([src_out, src_in, dst_out, dst_in], dtype=np.float32)

    # Node features include node type, node basis and HTTP evidence features.
    return np.concatenate([src, dst, np.abs(src - dst), relation, basis, topology])


X, y, metadata = [], [], []
for example in examples:
    sid = example.scenario_id
    edge = example.edge
    X.append(features_for_edge(
        graph_by_id[sid], edge.source, edge.target, edge.edge_type
    ))
    y.append(int(edge.label))
    metadata.append({
        "scenario_id": sid,
        "split": example.split,
        "source": edge.source,
        "target": edge.target,
        "edge_type": edge.edge_type,
        "label": int(edge.label),
    })

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int64)
train_idx = np.array([i for i, m in enumerate(metadata) if m["split"] == "train"])
val_idx = np.array([i for i, m in enumerate(metadata) if m["split"] == "validation"])
test_idx = np.array([i for i, m in enumerate(metadata) if m["split"] == "test"])

if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
    raise SystemExit(
        f"Empty split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )
if len(np.unique(y[train_idx])) < 2:
    raise SystemExit("Training split needs both positive and negative labels.")

# Fit preprocessing on training data only.
scaler = StandardScaler()
X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
X_val = scaler.transform(X[val_idx]).astype(np.float32)
X_test = scaler.transform(X[test_idx]).astype(np.float32)
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]


# -------------------- Bayesian neural network: Bayes by Backprop --------------------

class BayesianLinear(nn.Module):
    """Gaussian variational posterior over weights and biases."""

    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))
        self.prior_sigma = prior_sigma
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.zeros_(self.bias_mu)

    @staticmethod
    def sigma(rho):
        return torch.nn.functional.softplus(rho) + 1e-6

    def forward(self, x, sample=True):
        ws = self.sigma(self.weight_rho)
        bs = self.sigma(self.bias_rho)
        if sample:
            weight = self.weight_mu + ws * torch.randn_like(ws)
            bias = self.bias_mu + bs * torch.randn_like(bs)
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)

    def kl_divergence(self):
        # Analytic KL(q || p), where p is a zero-mean Gaussian prior.
        prior_var = self.prior_sigma ** 2
        def kl(mu, rho):
            sigma = self.sigma(rho)
            return (
                torch.log(torch.tensor(self.prior_sigma, device=mu.device) / sigma)
                + (sigma.square() + mu.square()) / (2.0 * prior_var)
                - 0.5
            ).sum()
        return kl(self.weight_mu, self.weight_rho) + kl(self.bias_mu, self.bias_rho)


class BayesianRiskNN(nn.Module):
    def __init__(self, input_dim, hidden=8):
        super().__init__()
        self.layer1 = BayesianLinear(input_dim, hidden)
        self.layer2 = BayesianLinear(hidden, 1)

    def forward(self, x, sample=True):
        x = torch.relu(self.layer1(x, sample=sample))
        return self.layer2(x, sample=sample).squeeze(-1)

    def kl_divergence(self):
        return self.layer1.kl_divergence() + self.layer2.kl_divergence()


model = BayesianRiskNN(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCEWithLogitsLoss()
xt = torch.tensor(X_train)
yt = torch.tensor(y_train, dtype=torch.float32)
xv = torch.tensor(X_val)
yv = torch.tensor(y_val, dtype=torch.float32)

best_state = None
best_val_loss = float("inf")
epochs_without_improvement = 0
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    logits = model(xt, sample=True)
    data_loss = loss_fn(logits, yt)
    # Normalize KL by the number of training examples.
    kl_loss = model.kl_divergence() / len(yt)
    loss = data_loss + kl_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    # Validation uses the posterior mean for a stable checkpoint criterion.
    model.eval()
    with torch.no_grad():
        val_logits = model(xv, sample=False)
        val_loss = loss_fn(val_logits, yv).item()

    history.append({
        "epoch": epoch,
        "train_objective": float(loss.item()),
        "validation_bce": float(val_loss),
    })
    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        break

if best_state is None:
    raise SystemExit("Training failed to produce a checkpoint.")
model.load_state_dict(best_state)


# -------------------- Deterministic baseline --------------------

baseline = LogisticRegression(C=0.1, max_iter=2000, random_state=SEED)
baseline.fit(X_train, y_train)
baseline_p = baseline.predict_proba(X_test)[:, 1]


# -------------------- Bayesian posterior predictive inference --------------------

model.eval()
test_tensor = torch.tensor(X_test)
draws = []
with torch.no_grad():
    for _ in range(SAMPLES):
        draws.append(torch.sigmoid(model(test_tensor, sample=True)).numpy())
draws = np.asarray(draws)

mean_p = draws.mean(axis=0)
std_p = draws.std(axis=0)
low = np.quantile(draws, 0.05, axis=0)
high = np.quantile(draws, 0.95, axis=0)


def metric_report(actual, probabilities):
    predicted = (probabilities >= 0.5).astype(int)
    report = {
        "n": int(len(actual)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(actual, probabilities)),
        "mae": float(mean_absolute_error(actual, probabilities)),
        "roc_auc": (
            float(roc_auc_score(actual, probabilities))
            if len(np.unique(actual)) == 2 else None
        ),
    }
    return report


def risk_band(p):
    if p < 0.33:
        return "LOW"
    if p < 0.67:
        return "MEDIUM"
    return "HIGH"


predictions = []
for j, idx in enumerate(test_idx):
    m = metadata[idx]
    predictions.append({
        **m,
        "deterministic_baseline_probability": round(float(baseline_p[j]), 5),
        "bayesian_mean_probability": round(float(mean_p[j]), 5),
        "bayesian_predictive_std": round(float(std_p[j]), 5),
        "bayesian_90_percent_interval": [
            round(float(low[j]), 5), round(float(high[j]), 5)
        ],
        "proxy_risk_band": risk_band(float(mean_p[j])),
    })

MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "state_dict": model.state_dict(),
    "input_dim": int(X.shape[1]),
    "hidden_dim": 8,
    "prior_sigma": 1.0,
    "feature_scaler_mean": scaler.mean_.tolist(),
    "feature_scaler_scale": scaler.scale_.tolist(),
    "seed": SEED,
}, MODEL_OUT)

result = {
    "experiment": "phase_16_bayes_by_backprop",
    "seed": SEED,
    "method": "Variational Bayesian neural network with Gaussian weight posteriors",
    "prior": "Independent zero-mean Gaussian weights, sigma=1.0",
    "training": {
        "epochs_run": len(history),
        "best_validation_bce": best_val_loss,
        "train_edges": int(len(train_idx)),
        "validation_edges": int(len(val_idx)),
        "test_edges": int(len(test_idx)),
        "train_scenarios": int(len({metadata[i]["scenario_id"] for i in train_idx})),
        "validation_scenarios": int(len({metadata[i]["scenario_id"] for i in val_idx})),
        "test_scenarios": int(len({metadata[i]["scenario_id"] for i in test_idx})),
        "input_feature_count": int(X.shape[1]),
        "posterior_samples_at_test": SAMPLES,
    },
    "deterministic_baseline": {
        "model": "Regularised logistic regression on the same graph-derived features",
        "test_metrics": metric_report(y_test, baseline_p),
    },
    "bayesian_model": {
        "test_metrics": metric_report(y_test, mean_p),
        "predictions": predictions,
    },
    "interpretation": {
        "score": "Mean posterior predictive probability for a positive labelled edge assessment.",
        "uncertainty": "Standard deviation and 5th-95th percentile interval across sampled Bayesian weights.",
        "risk_bands": "LOW < 0.33; MEDIUM 0.33-0.67; HIGH >= 0.67. Prototype thresholds, not operationally validated.",
        "limitation": (
            "This estimates an attack-edge-label proxy, not post-remediation residual risk. "
            "The dataset is very small; uncertainty intervals are not established as calibrated. "
            "Use independent post-remediation labels and calibration evaluation before operational use."
        ),
    },
    "training_history": history,
}

OUT.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({
    "saved_results": str(OUT),
    "saved_model": str(MODEL_OUT),
    "training": result["training"],
    "baseline_test_metrics": result["deterministic_baseline"]["test_metrics"],
    "bayesian_test_metrics": result["bayesian_model"]["test_metrics"],
    "sample_predictions": predictions[:5],
    "caveat": result["interpretation"]["limitation"],
}, indent=2))
