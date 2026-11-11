"""Classification heads on frozen features.

Rep      : standardize -> [LDA(1) + PCA(q-1)] (or PCA-q) -> angle scaling to [0, pi]; every step fit on TRAINING rows only and
           folded into one affine map so the whole path stays differentiable (needed for Grad-CAM through the head).
VQC      : noiseless statevector simulator in pure PyTorch (no shots, no hardware). RY(angle) encoding, L layers of
           per-qubit Rot = RZ RY RZ followed by a CNOT entangler ('pairs' brick pattern or 'ring'), readout <Z_0>, logit = a<Z_0> + b.
LinHead  : logistic regression head on the same inputs (classical control / federated classical head).
"""
import math, numpy as np, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

torch.set_default_dtype(torch.float32)


class Rep:
    def __init__(self, q=12, kind="lda_pca"):
        self.q, self.kind = q, kind

    def fit(self, X, y):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6; Z = (X - self.mu) / self.sd
        if self.kind == "pca":
            p = PCA(self.q, svd_solver="randomized", random_state=0).fit(Z); self.W = p.components_.T.copy(); self.b = -p.mean_ @ self.W
        else:
            lda = LDA(solver="eigen", shrinkage="auto", n_components=1).fit(Z, y); p = PCA(self.q - 1, svd_solver="randomized", random_state=0).fit(Z)
            W1 = lda.scalings_[:, :1]; b1 = np.zeros(1);                  # 'eigen' LDA applies no centering; min-max scaling below removes any offset
            self.W = np.hstack([W1, p.components_.T]); self.b = np.concatenate([b1, -p.mean_ @ p.components_.T])
        T = Z @ self.W + self.b; self.lo = T.min(0); self.hi = T.max(0); return self

    def transform(self, X):                                                     # numpy -> angles in [0, pi]
        T = ((X - self.mu) / self.sd) @ self.W + self.b; return np.clip(math.pi * (T - self.lo) / np.where(self.hi - self.lo < 1e-9, 1.0, self.hi - self.lo), 0, math.pi).astype(np.float32)

    def torch_params(self):                                                     # for differentiable use: x -> angles
        t = lambda a: torch.tensor(a, dtype=torch.float32); return dict(mu=t(self.mu), sd=t(self.sd), W=t(self.W), b=t(self.b), lo=t(self.lo), hi=t(np.where(self.hi - self.lo < 1e-9, 1.0, self.hi - self.lo)))


def torch_rep(x, P):
    T = ((x - P["mu"]) / P["sd"]) @ P["W"] + P["b"]; return torch.clamp(math.pi * (T - P["lo"]) / P["hi"], 0, math.pi)   # P["hi"] holds (hi - lo)


def _cnot_perm(n, c, t):
    idx = np.arange(2 ** n); cbit = (idx >> (n - 1 - c)) & 1; return idx ^ (cbit << (n - 1 - t))


def entangler_perm(n, kind):
    if kind == "none": return np.arange(2 ** n)
    pairs = [(c, (c + 1) % n) for c in range(n)] if kind == "ring" else [(c, c + 1) for c in range(0, n - 1, 2)] + [(c, c + 1) for c in range(1, n - 1, 2)]
    comp = np.arange(2 ** n)
    for c, t in pairs: comp = comp[_cnot_perm(n, c, t)]                           # state' = state[p]; compositions index into each other
    return comp


class VQC(nn.Module):
    def __init__(self, n=12, L=2, ent="pairs", seed=0, init_std=0.5):
        super().__init__(); g = torch.Generator().manual_seed(seed); self.n, self.L, self.ent = n, L, ent
        self.theta = nn.Parameter(init_std * torch.randn(L, n, 3, generator=g)); self.a = nn.Parameter(torch.tensor(2.0)); self.b = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("perm", torch.from_numpy(entangler_perm(n, ent)).long()); idx = torch.arange(2 ** n); self.register_buffer("z0", 1.0 - 2.0 * ((idx >> (n - 1)) & 1).float())

    @staticmethod
    def _rot(phi, th, om):                                                       # Rot = RZ(om) RY(th) RZ(phi), shape [...,2,2] complex
        c, s = torch.cos(th / 2), torch.sin(th / 2); e = lambda a: torch.polar(torch.ones_like(a), a)
        m00 = e(-(phi + om) / 2) * c; m01 = -e((phi - om) / 2) * s; m10 = e(-(phi - om) / 2) * s; m11 = e((phi + om) / 2) * c
        return torch.stack([torch.stack([m00, m01], -1), torch.stack([m10, m11], -1)], -2)

    def state(self, x):                                                          # x: [B, n] angles -> [B, 2^n] complex state
        B, n = x.shape; s = torch.ones(B, 1)
        for q in range(n): s = (s.unsqueeze(2) * torch.stack([torch.cos(x[:, q] / 2), torch.sin(x[:, q] / 2)], 1).unsqueeze(1)).reshape(B, -1)
        s = s.to(torch.complex64)
        for l in range(self.L):
            U = self._rot(self.theta[l, :, 0], self.theta[l, :, 1], self.theta[l, :, 2])                  # [n,2,2]
            for q in range(n): s = torch.einsum("ij,bajk->baik", U[q], s.reshape(B, 2 ** q, 2, 2 ** (n - q - 1))).reshape(B, -1)
            s = s[:, self.perm]
        return s

    def expz(self, x):
        s = self.state(x); return (s.real ** 2 + s.imag ** 2) @ self.z0

    def forward(self, x):
        return self.a * self.expz(x) + self.b

    def n_params(self):
        return self.theta.numel() + 2


class LinHead(nn.Module):
    def __init__(self, d, seed=0):
        super().__init__(); g = torch.Generator().manual_seed(seed); self.w = nn.Parameter(0.1 * torch.randn(d, generator=g)); self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):                                                        # centered angles keep the logit scale comparable
        return (x - math.pi / 2) @ self.w + self.b

    def n_params(self):
        return self.w.numel() + 1


def class_weights(y):
    y = np.asarray(y); w = np.where(y == 1, 0.5 / max(y.mean(), 1e-9), 0.5 / max(1 - y.mean(), 1e-9)); return torch.tensor(w, dtype=torch.float32)


def fit_head(model, Xa, y, lr=0.05, epochs=60, batch=128, seed=0, Xv=None, yv=None, patience=12, weight_decay=0.0, return_curve=False):
    """AdamW + weighted BCE. If a validation set is given, keeps the best-validation-AUC weights and reports the best epoch."""
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(seed); rng = np.random.default_rng(seed); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    X = torch.from_numpy(Xa); Y = torch.from_numpy(np.asarray(y, np.float32)); W = class_weights(y); best, best_ep, best_state, bad = -1.0, 0, None, 0; curve = []
    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(X)); model.train()
        for i in range(0, len(X), batch):
            ix = perm[i:i + batch]; opt.zero_grad(); loss = (nn.functional.binary_cross_entropy_with_logits(model(X[ix]), Y[ix], reduction="none") * W[ix]).mean(); loss.backward(); opt.step()
        if Xv is not None:
            with torch.no_grad(): pv = torch.sigmoid(model(torch.from_numpy(Xv))).numpy()
            auc = roc_auc_score(yv, pv); curve.append(auc)
            if auc > best + 1e-4: best, best_ep, bad = auc, ep, 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return (best, best_ep, curve) if return_curve else (best, best_ep)


@torch.no_grad()
def predict(model, Xa, batch=512):
    model.eval(); out = []
    for i in range(0, len(Xa), batch): out.append(torch.sigmoid(model(torch.from_numpy(Xa[i:i + batch]))).numpy())
    return np.concatenate(out)


def selftest():
    import json
    res = {}; n = 6; m = VQC(n, 2, "pairs", seed=1); x = torch.rand(5, n) * math.pi; s = m.state(x); res["norm_error"] = float((s.real ** 2 + s.imag ** 2).sum(1).sub(1).abs().max())
    # Bell test: H-free check using RY(pi/2) on qubit 0 then CNOT(0->1): <Z0 Z1> = 1, <Z0> = 0
    st = torch.zeros(1, 4); st[0, 0] = math.cos(math.pi / 4); st[0, 2] = math.sin(math.pi / 4); st = st.to(torch.complex64); p = torch.from_numpy(_cnot_perm(2, 0, 1)).long(); st = st[:, p]; pr = (st.real ** 2 + st.imag ** 2)[0]
    res["bell_ZZ"] = float(pr[0] + pr[3] - pr[1] - pr[2]); res["bell_Z0"] = float(pr[0] + pr[1] - pr[2] - pr[3])
    # closed form for the product circuit (no entangler): <Z0> from 2x2 algebra
    m0 = VQC(n, 2, "none", seed=2); xs = torch.rand(3, n) * math.pi; ez = m0.expz(xs)
    cf = []
    for r in range(3):
        v = torch.tensor([math.cos(float(xs[r, 0]) / 2), math.sin(float(xs[r, 0]) / 2)], dtype=torch.complex64)
        for l in range(2): v = VQC._rot(m0.theta[l, 0, 0], m0.theta[l, 0, 1], m0.theta[l, 0, 2]).detach() @ v
        cf.append(float((v.abs() ** 2)[0] - (v.abs() ** 2)[1]))
    res["closed_form_error"] = float((ez - torch.tensor(cf)).abs().max())
    xg = (torch.rand(3, n) * math.pi).requires_grad_(True); m(xg).sum().backward(); res["grad_finite"] = bool(torch.isfinite(xg.grad).all())
    res["n_params_12q_3L"] = VQC(12, 3).n_params(); return res


if __name__ == "__main__":
    import json; print(json.dumps(selftest(), indent=1))
