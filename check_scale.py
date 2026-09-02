"""
DO PHAN UNG VOI THANG DO INPUT  —  hopfield_models.HopfieldEnergySolver
=======================================================================
Chay tren dung cau hinh HIEN TAI cua file (norm_x da comment hay chua deu duoc).
KHONG sua model. Chi do.

Hai dai luong, do cung mot luc:

  1. d_hat(c) = dlog||f(c x)|| / dlog c        <- bac dong nhat HIEU DUNG
     Task can d = +1 (addition/lowrank) hoac -1 (inverse).
     d_hat on dinh dung bang d o MOI c  =>  dong nhat chinh xac.
     d_hat troi                          =>  khong dong nhat.

  2. max_j A_ij  va  entropy(A_i)              <- DO SAC cua attention
     Day la phep kiem CO CHE, khong phai trieu chung:
     neu beta_eff ~ c^2 thi softmax sac len -> max_A -> 1.0 va entropy -> 0.
     Voi Addition, dap an dung can HAI key moi cai 0.5, tuc max_A ~ 0.5.
     max_A -> 1.0 nghia la model chi con copy MOT so hang, khong cong duoc.

Chay:  python check_scale.py
"""

import math
import torch
from hopfield_models import HopfieldEnergySolver as M

torch.manual_seed(0)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DT = torch.float64
STEPS = 20
CS = [1.0, 2.5, 10.0, 100.0, 1e3, 1e4]


def attn_stats(m, z, x_tokens):
    """Tra (max_A trung binh, entropy trung binh) o buoc hien tai."""
    u = x_tokens                                  # dung dung cai model dang dung
    if 'self.norm_x(x_tokens)' in getattr(m.forward_step, '__doc__', '') or False:
        pass
    Q = m.W_q(z)
    K = m.W_k(u)
    B = Q.size(0)
    d_h = m.d_k // m.num_heads
    Qh = Q.view(B, -1, m.num_heads, d_h).transpose(1, 2)
    Kh = K.view(B, -1, m.num_heads, d_h).transpose(1, 2)
    scale = (m.beta / math.sqrt(d_h)) if m.learnable_beta else 1.0 / math.sqrt(d_h)
    A = torch.softmax(Qh @ Kh.transpose(-2, -1) * scale, dim=-1)
    maxA = A.max(dim=-1).values.mean().item()
    ent = (-(A.clamp_min(1e-300) * A.clamp_min(1e-300).log()).sum(dim=-1)).mean().item()
    return maxA, ent


def run(inp_dim, out_dim, d_target, tie_mode, label):
    torch.manual_seed(0)
    m = M(inp_dim, out_dim, num_heads=8, step_lr=0.5,
          beta=None, tie_mode=tie_mode).to(DT).to(DEV).eval()
    x0 = (torch.rand(4, inp_dim, dtype=DT, device=DEV) - 0.5) * 2
    y0 = (torch.rand(4, out_dim, dtype=DT, device=DEV) - 0.5) * 2

    norms, maxAs, ents = [], [], []
    with torch.no_grad():
        for c in CS:
            xt = m.embed_input(c * x0)
            z = m.embed_latent(y0)
            for _ in range(STEPS):
                z = m.forward_step(z, xt)
            norms.append(m.decode(z).norm().item())
            a, e = attn_stats(m, z, xt)
            maxAs.append(a); ents.append(e)

    print(f"\n### {label}   tie={tie_mode}   d can = {d_target:+.0f} ###")
    hdr = "".join(f"{c:>11.0e}" for c in CS)
    print(f"    c        {hdr}")
    print(f"    ||f||    " + "".join(f"{v:>11.3e}" for v in norms))
    dh = ["      —    "]
    for i in range(1, len(CS)):
        dh.append(f"{math.log(norms[i]/max(norms[i-1],1e-300))/math.log(CS[i]/CS[i-1]):>+11.3f}")
    print(f"    d_hat    " + "".join(dh))
    print(f"    max_A    " + "".join(f"{v:>11.4f}" for v in maxAs))
    print(f"    entropy  " + "".join(f"{v:>11.4f}" for v in ents))

    drift = max(abs(float(t) - d_target) for t in
                [x.strip() for x in dh[1:]])
    sharp = maxAs[-1] / max(maxAs[0], 1e-30)
    print(f"    -> lech d_hat lon nhat = {drift:.3f}"
          f"   |   max_A tang {sharp:.2f}x tu c=1 den c=1e4")
    if drift < 0.05:
        print("    => DONG NHAT (d_hat on dinh) — khong can gauge")
    else:
        print("    => KHONG dong nhat — d_hat troi")
    if sharp > 1.5:
        print("    => ATTENTION SAC LEN theo thang do -> landscape bi meo (co che §1.6-A)")


if __name__ == "__main__":
    print("=" * 90)
    print("PHAN UNG VOI THANG DO INPUT — cau hinh HIEN TAI cua hopfield_models.py")
    print("Doc: d_hat on dinh = dong nhat.  max_A tang = attention sac len = landscape doi.")
    print("Voi Addition, dap an dung can max_A ~ 0.5 (hai key). max_A -> 1.0 la mat kha nang cong.")
    print("=" * 90)
    run(800, 400, +1.0, 'hard', "ADDITION")
    run(400, 400, -1.0, 'hard', "INVERSE ")
    run(800, 400, +1.0, 'random', "ADDITION")
    print("\n" + "=" * 90)
