"""
KIEM tie_mode  —  hopfield_models.HopfieldEnergySolver
======================================================
    E = -1/beta * sum_h LSE_h( <(z W_q)^h, (u W_k)^h> )  +  0.5*||z||^2      u = norm_x(X)
    z <- (1 - lam) * z  +  lam * attn_out

tie_mode='hard' :  attn_out = sum_h A^h @ ( K^h (W_q^h)^T )
                   == -grad(E1) CHINH XAC  ->  E PHAI giam don dieu (0 vi pham)
tie_mode='random' :  attn_out = concat_h( A^h @ V^h ) W_o,  V = W_v(u)
                   CHUA bang -grad(E1) khi chua tie  ->  con vi pham don dieu
                   tie_penalty() do do lech; ~2.0 luc init, 0.0 khi tie hoan hao

PHAN 3 phan tich CAN BANG LOSS: so sanh CHUAN GRADIENT cua MSE va cua R (khong so
sanh gia tri loss — hai loss co the lech gia tri nhieu ma gradient tuong duong).

Chay:  python check_descent.py
"""

import torch
from hopfield_models import HopfieldEnergySolver as M

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DT = torch.float64
STEPS = 80
RES = []


def check(name, ok, detail="", critical=True):
    RES.append(ok or not critical)
    tag = 'PASS' if ok else ('FAIL' if critical else 'warn')
    print(f"    [{tag}] {name:<42s} {detail}")


# ============================================================ PHAN 1+2: descent
def run(inp_dim, out_dim, num_heads, lam, tie_mode, label, strict_mono):
    torch.manual_seed(0)
    m = M(inp_dim, out_dim, num_heads=num_heads, step_lr=lam,
          beta=None, tie_mode=tie_mode).to(DT).to(DEV).eval()
    npar = sum(p.numel() for p in m.parameters())
    x = (torch.rand(8, inp_dim, dtype=DT, device=DEV) - 0.5) * 2
    y0 = (torch.rand(8, out_dim, dtype=DT, device=DEV) - 0.5) * 2

    with torch.no_grad():
        xt = m.embed_input(x)
        z = m.embed_latent(y0)
        E, zn, dzn = [], [], []
        for _ in range(STEPS):
            zp = z
            z = m.forward_step(z, xt)
            E.append(m.get_energy(z, xt).mean().item())
            zn.append(z.norm(dim=-1).mean().item())
            dzn.append((z - zp).norm(dim=-1).mean().item())

    print(f"\n### {label}  tie={tie_mode}  lam={lam}  params={npar:,} ###")
    f = lambda v: "  ".join(f"{t:9.3f}" for t in v)
    print(f"    E    : {f(E[:5])}  ...  {f(E[-2:])}")
    print(f"    |z|  : {f(zn[:5])}  ...  {f(zn[-2:])}")
    print(f"    |dz| : {f(dzn[:5])}  ...  {f(dzn[-2:])}")

    ratio = dzn[-1] / max(zn[-1], 1e-30)
    check("||dz||/||z|| -> 0", ratio < 1e-6, f"= {ratio:.3e}")

    viol = sum(1 for i in range(1, len(E)) if E[i] > E[i - 1] + 1e-9)
    check("E giam don dieu", viol == 0, f"vi pham = {viol}/{STEPS-1}", critical=strict_mono)
    if not strict_mono:
        print(f"           ^ soft chua tie -> con vi pham la DUNG NHU MONG DOI")


# ============================================================ PHAN 3: can bang loss
def balance(inp_dim=800, out_dim=400, num_heads=8, B=64):
    print("\n" + "=" * 78)
    print("PHAN 3 — CAN BANG GIUA MSE VA tie_penalty  (tie_mode='random')")
    print("=" * 78)
    torch.manual_seed(0)
    m = M(inp_dim, out_dim, num_heads=num_heads, step_lr=0.5,
          beta=None, tie_mode='random').to(DT).to(DEV)

    # mot batch Addition gia lap
    x = (torch.rand(B, inp_dim, dtype=DT, device=DEV) - 0.5) * 2
    y = x[:, :out_dim] + x[:, out_dim:]
    y0 = (torch.rand(B, out_dim, dtype=DT, device=DEV) - 0.5) * 2

    def fwd():
        xt = m.embed_input(x)
        z = m.embed_latent(y0)
        for _ in range(5):
            z = m.forward_step(z, xt)
        return ((m.decode(z) - y) ** 2).mean()

    shared = [m.W_q.weight, m.W_k.weight, m.W_v.weight, m.W_o.weight]

    mse = fwd()
    g_mse = torch.autograd.grad(mse, shared, retain_graph=False, allow_unused=True)
    n_mse = torch.stack([g.norm() for g in g_mse if g is not None]).norm().item()

    R = m.tie_penalty()
    g_R = torch.autograd.grad(R, shared, allow_unused=True)
    n_R = torch.stack([g.norm() for g in g_R if g is not None]).norm().item()

    print(f"\n  GIA TRI loss  :  MSE = {mse.item():.6e}     R = {R.item():.6e}")
    print(f"  CHUAN GRADIENT:  |dMSE| = {n_mse:.6e}   |dR| = {n_R:.6e}")
    print(f"  ty le |dR| / |dMSE| = {n_R / max(n_mse, 1e-30):.4e}")

    for frac, lab in [(1.0, "ngang bang"), (0.1, "R la 10% cua MSE"), (0.01, "R la 1%")]:
        g = frac * n_mse / max(n_R, 1e-30)
        print(f"    gamma de {lab:<18s} = {g:.4e}")

    check("R ~ 2.0 luc init ngau nhien", 1.0 < R.item() < 4.0, f"R = {R.item():.4f}")
    print("""
  CACH DOC:
    * R da CHUAN HOA nen ~2.0 luc init va 0.0 khi tie -> gamma so sanh duoc giua cac config.
    * DUNG chon gamma theo GIA TRI loss, chon theo CHUAN GRADIENT (dong 'gamma de ...').
    * KHONG di tim "gamma dung nhat". Muc dich cua soft KHONG phai dua R ve 0
      (hard da co R = 0 san, va it hon 2*d_model^2 tham so). Muc dich la xem 4096
      chieu gauge tu do co mua them do chinh xac hay khong.
      -> QUET gamma in {0.01, 0.1, 1, 10} roi ve duong doi lap (MSE cuoi, R cuoi).
         Duong do CHINH LA ket qua, khong phai mot con so gamma.
    * PHAI log R trong luc train. R di 2.0 -> 0.01 la bang chung ham phat co tac dung;
      R ket o 1.5 la gamma qua nho hoac rang buoc dang danh nhau voi MSE.""")


if __name__ == "__main__":
    print("=" * 78)
    print("PHAN 1 — tie_mode='hard':  E PHAI giam don dieu (0 vi pham)")
    print("=" * 78)
    for lam in [0.5, 0.25]:
        run(800, 400, 8, lam, 'hard', "ADDITION", strict_mono=True)
    run(400, 400, 8, 0.5, 'hard', "INVERSE ", strict_mono=True)
    run(800, 400, 1, 0.5, 'hard', "ADDITION H=1", strict_mono=True)

    print("\n" + "=" * 78)
    print("PHAN 2 — tie_mode='random' chua tie:  con vi pham la DUNG")
    print("=" * 78)
    run(800, 400, 8, 0.5, 'random', "ADDITION", strict_mono=False)

    balance()

    print("\n" + "=" * 78)
    print(f"KET QUA (muc bat buoc): {sum(RES)}/{len(RES)} pass")
    print("=" * 78)
