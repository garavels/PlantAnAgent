"""Forward saved pilot tensors on recorded observations without training/reset."""

from pathlib import Path
import io, json, gzip, zipfile
import torch, numpy as np
from torch.nn import functional as F

P = Path("/Volumes/Lexar/PlantAnAgentResearch/pilot-pi-seed101-v1")
torch.set_num_threads(2)
initial = torch.load(
    P / "artifacts/initial-policy.pt", weights_only=True, map_location="cpu"
)
with zipfile.ZipFile(P / "artifacts/final-policy.zip") as z:
    final = torch.load(
        io.BytesIO(z.read("policy.pth")), weights_only=True, map_location="cpu"
    )


def mlp(x, ps, prefix, ixs):
    for i in ixs:
        x = torch.tanh(
            F.linear(x, ps[f"{prefix}.{i}.weight"], ps[f"{prefix}.{i}.bias"])
        )
    return x


@torch.no_grad()
def infer(obs, ps):
    obs = torch.tensor(obs, dtype=torch.float32)
    pi = mlp(obs, ps, "mlp_extractor.policy_net", [0, 2])
    v = mlp(obs, ps, "mlp_extractor.value_net", [0, 2])
    mu = F.linear(pi, ps["action_net.weight"], ps["action_net.bias"])
    values = F.linear(v, ps["value_net.weight"], ps["value_net.bias"]).flatten()
    cs = []
    for k in range(4):
        c = mlp(obs, ps, f"cost_critic.heads.{k}", [0, 2])
        cs.append(
            F.linear(
                c,
                ps[f"cost_critic.heads.{k}.4.weight"],
                ps[f"cost_critic.heads.{k}.4.bias"],
            ).flatten()
        )
    normal = torch.distributions.Normal(mu, ps["log_std"].exp())
    return (
        mu.numpy(),
        values.numpy(),
        torch.stack(cs, axis=1).numpy(),
        normal.cdf(torch.full_like(mu, -1)).numpy(),
        (1 - normal.cdf(torch.full_like(mu, 1))).numpy(),
        (v.abs() > 0.99).float().mean().item(),
    )


out = {}
for actor, ps in [("initial", initial), ("final", final)]:
    runs = []
    for seed in range(9800000, 9800004):
        d = json.loads(
            gzip.decompress((P / f"evaluation/{seed}-{actor}.json.gz").read_bytes())
        )
        hh = d["history"]
        obs = [r["observation"] for r in hh]
        r = np.array([x["reward_eur"] for x in hh])
        cost = np.array(
            [x["info"]["cycle_constraint_transition"]["training_cost"] for x in hh]
        )
        a = np.array([x["action"] for x in hh])
        mu, v, cs, pl, ph, sat = infer(obs, ps)
        rt = np.zeros(len(r))
        acc = 0.0
        for i in range(len(r) - 1, -1, -1):
            acc = r[i] + 0.995 * acc
            rt[i] = acc
        td = r + 0.995 * np.r_[v[1:], 0] - v
        adv = np.zeros(len(r))
        acc = 0
        for i in range(len(r) - 1, -1, -1):
            acc = td[i] + 0.995 * 0.95 * acc
            adv[i] = acc
        cv = np.zeros_like(cost)
        acc = np.zeros(4)
        for i in range(len(r) - 1, -1, -1):
            acc = cost[i] + acc
            cv[i] = acc
        ctd = cost + np.vstack([cs[1:], np.zeros(4)]) - cs
        ca = np.zeros_like(ctd)
        acc = np.zeros(4)
        for i in range(len(r) - 1, -1, -1):
            acc = ctd[i] + 0.95 * acc
            ca[i] = acc
        runs.append(
            {
                "seed": seed,
                "action_prediction_max_abs_error": float(
                    np.max(np.abs(np.clip(mu, -1, 1) - a))
                ),
                "mu_mean": mu.mean(axis=0).tolist(),
                "mu_std": mu.std(axis=0).tolist(),
                "mu_min": mu.min(axis=0).tolist(),
                "mu_max": mu.max(axis=0).tolist(),
                "expected_lower_clipping": pl.mean(axis=0).tolist(),
                "expected_upper_clipping": ph.mean(axis=0).tolist(),
                "value_min_mean_max": [float(v.min()), float(v.mean()), float(v.max())],
                "discounted_return_min_mean_max": [
                    float(rt.min()),
                    float(rt.mean()),
                    float(rt.max()),
                ],
                "value_mse": float(np.mean((rt - v) ** 2)),
                "value_explained_variance": float(1 - np.var(rt - v) / np.var(rt)),
                "value_hidden_saturation_fraction": sat,
                "reward_gae_std": float(adv.std()),
                "cost_gae_std": ca.std(axis=0).tolist(),
                "cost_gae_mean": ca.mean(axis=0).tolist(),
                "cost_value_pred_mean": cs.mean(axis=0).tolist(),
                "undiscounted_remaining_cost_mean": cv.mean(axis=0).tolist(),
            }
        )
    out[actor] = {"gaussian_std": ps["log_std"].exp().tolist(), "episodes": runs}
Path("/private/tmp/plant-pilot-network-summary.json").write_text(
    json.dumps(out, indent=2) + "\n"
)
print(json.dumps(out, indent=2))
