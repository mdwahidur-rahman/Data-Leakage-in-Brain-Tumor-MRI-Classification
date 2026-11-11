"""Frozen pretrained-backbone feature extraction (global-average-pooled / CLS features).

usage: python features.py <backbone> <variant>      e.g. python features.py resnet50 full
Weights are downloaded from the GitHub release URLs of the original timm model zoo; the first 8 hex digits of the
SHA-256 of every downloaded file must match the digits embedded in its file name (timm convention) when present.
Resumable: features are written in chunks; an existing complete output is skipped.
"""
import os, sys, time, hashlib, numpy as np, torch, timm

torch.set_num_threads(int(os.environ.get("NT", 1)))
D = "./data"; F = "./feats"; os.makedirs(F, exist_ok=True)
GH = "https://github.com/rwightman/pytorch-image-models/releases/download/"
IMN = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)); HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
BB = {
    "resnet50": dict(arch="resnet50", url=GH + "v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth", norm=IMN),
    "densenet121": dict(arch="densenet121", url=GH + "v0.1-weights/densenet121_ra-50efcf5c.pth", norm=IMN),
    "inceptionv3": dict(arch="inception_v3", url=GH + "v0.1-weights/tf_inception_v3-e0069de4.pth", norm=HALF),
    "xception": dict(arch="legacy_xception", url=GH + "v0.1-cadene/xception-43020ad28.pth", norm=HALF),
    "vit": dict(arch="vit_base_patch16_224", url=GH + "v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth", norm=HALF, custom_load=False),
}
if os.path.exists("./weights/deit_small_patch16_224-cd65a155.pth"):       # added only if the user supplies the file
    BB["deit"] = dict(arch="deit_small_patch16_224", file="./weights/deit_small_patch16_224-cd65a155.pth", norm=IMN)


def fetch(url):
    os.makedirs("./weights", exist_ok=True); p = "./weights/" + url.split("/")[-1]
    if not os.path.exists(p):
        import urllib.request
        for attempt in range(6):
            try:
                urllib.request.urlretrieve(url, p + ".part"); os.replace(p + ".part", p); break
            except Exception as e:
                print("download retry", attempt, str(e)[:60], flush=True); time.sleep(5 * (attempt + 1))
    tag = os.path.basename(p).rsplit("-", 1)[-1].split(".")[0] if "-" in os.path.basename(p) else None
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if tag and len(tag) == 8 and all(c in "0123456789abcdef" for c in tag): assert h.startswith(tag), f"checksum mismatch for {p}: {h[:8]} != {tag}"
    return p, h[:12]


def build(name):
    c = BB[name]; path, sha = (c["file"], "user-supplied") if "file" in c else fetch(c["url"])
    kw = dict(pretrained=True, num_classes=0, pretrained_cfg_overlay=dict(file=path, url=None, hf_hub_id=None, custom_load=c.get("custom_load", False)))
    m = timm.create_model(c["arch"], **kw).eval(); return m, c["norm"], sha


@torch.no_grad()
def extract(name, variant, batch=16):
    out = f"{F}/{name}_{variant}.npy"
    if os.path.exists(out): print("exists", out); return
    m, (mean, std), sha = build(name); X = np.load(f"{D}/images_{variant}.npy", mmap_mode="r"); n = len(X); mean = torch.tensor(mean).view(1, 3, 1, 1); std = torch.tensor(std).view(1, 3, 1, 1)
    CH = 320; cdir = out + ".chunks"; os.makedirs(cdir, exist_ok=True); t0 = time.time(); nch = (n + CH - 1) // CH
    for ci, s0 in enumerate(range(0, n, CH)):
        cf = f"{cdir}/{ci:04d}.npy"
        if os.path.exists(cf): continue
        feats = []
        for i in range(s0, min(n, s0 + CH), batch):
            x = torch.from_numpy(np.ascontiguousarray(X[i:min(i + batch, s0 + CH, n)])).float().div(255.0).unsqueeze(1).repeat(1, 3, 1, 1); x = (x - mean) / std
            feats.append(m(x).numpy().astype(np.float32))
        np.save(cf + ".tmp.npy", np.concatenate(feats)); os.replace(cf + ".tmp.npy", cf); print(f"{name}/{variant} chunk {ci + 1}/{nch}  {time.time() - t0:.0f}s", flush=True)
    allf = np.concatenate([np.load(f"{cdir}/{i:04d}.npy") for i in range(nch)]); assert allf.shape[0] == n
    np.save(out + ".tmp.npy", allf); os.replace(out + ".tmp.npy", out); open(out + ".sha", "w").write(f"weights sha256[:12]={sha}\n"); print("saved", out, allf.shape, flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3: extract(sys.argv[1], sys.argv[2])
    else:                                                             # smoke test: every backbone builds, loads strictly, and runs
        for k in BB:
            t = time.time(); m, (mean, std), sha = build(k); x = torch.randn(4, 3, 224, 224)
            with torch.no_grad(): y = m(x)
            t1 = time.time(); 
            with torch.no_grad(): m(x)
            print(f"{k:12s} OK  feat_dim={y.shape[1]}  sha={sha}  {(time.time() - t1) / 4:.3f} s/img", flush=True)
