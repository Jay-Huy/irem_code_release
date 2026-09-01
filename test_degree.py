"""
ĐO BẬC ĐỒNG NHẤT HIỆU DỤNG  d_hat(c) = dlog||f(c x)|| / dlog c
===============================================================
So sánh 4 biến thể trên cùng một bộ weight:

  V1  current      : V = W_v(x_tokens)                 <- code hiện tại (Decoupled Norm)
  V2  V-normalized : V = W_v(norm_x(x_tokens))         <- chuẩn hóa cả nhánh V
  V3  strict-cons  : V = K @ W_q.weight, bỏ W_v, W_o   <- gradient Hopfield ĐÚNG (conservative)
  V4  wrapper      : V1 + chuẩn hóa instance, nhân r^d

Kỳ vọng:  V1 -> d_hat tiệm cận +1 (mọi task)
          V2, V3 -> d_hat ~ 0  (trạng thái bị chặn: KHÔNG ngoại suy được)
          V4 -> d_hat = d đúng bằng, với mọi c

Chạy:  python test_degree.py
"""

import math
import torch
import numpy as np
from hopfield_models import HopfieldEnergySolver

torch.manual_seed(0); np.random.seed(0)
DEV   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
STEPS, LAM = 10, 0.5
CS = [1.0, 2.5, 10.0, 100.0, 1e3, 1e4, 1e5]


def _heads(t, n, dh):
    return t.view(t.size(0), -1, n, dh).transpose(1, 2)


def _scale(m, dkh):
    return (m.beta / np.sqrt(dkh)) if m.learnable_beta else 1.0 / np.sqrt(dkh)


def step_current(m, z, xt, lam):
    """V = W_v(x_tokens)  — code hiện tại."""
    Q, K, V = m.W_q(m.norm_z(z)), m.W_k(m.norm_x(xt)), m.W_v(xt)
    dkh, dvh = m.d_k // m.num_heads, m.d_model // m.num_heads
    Q, K, V = _heads(Q, m.num_heads, dkh), _heads(K, m.num_heads, dkh), _heads(V, m.num_heads, dvh)
    a = torch.softmax(Q @ K.transpose(-2, -1) * _scale(m, dkh), dim=-1)
    o = (a @ V).transpose(1, 2).contiguous().view(z.size(0), -1, m.d_model)
    return (1.0 - lam) * z + lam * m.W_o(o)


def step_normV(m, z, xt, lam):
    """V = W_v(norm_x(x_tokens)) — chỉ đổi đúng một chỗ."""
    xn = m.norm_x(xt)
    Q, K, V = m.W_q(m.norm_z(z)), m.W_k(xn), m.W_v(xn)
    dkh, dvh = m.d_k // m.num_heads, m.d_model // m.num_heads
    Q, K, V = _heads(Q, m.num_heads, dkh), _heads(K, m.num_heads, dkh), _heads(V, m.num_heads, dvh)
    a = torch.softmax(Q @ K.transpose(-2, -1) * _scale(m, dkh), dim=-1)
    o = (a @ V).transpose(1, 2).contiguous().view(z.size(0), -1, m.d_model)
    return (1.0 - lam) * z + lam * m.W_o(o)


def step_strict(m, z, xt, lam):
    """Gradient Hopfield ĐÚNG, 1 head: V = K W_q^T, không W_v, không W_o."""
    Q = m.W_q(m.norm_z(z))                 # (B, out, d_k)
    K = m.W_k(m.norm_x(xt))                # (B, inp, d_k)
    V = K @ m.W_q.weight                   # (B, inp, d_model)   <- V = K W_q^T
    a = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(m.d_k), dim=-1)
    return (1.0 - lam) * z + lam * (a @ V)


def make_solver(step_fn):
    def solve(m, x):
        xt = m.embed_input(x)
        y0 = torch.zeros(x.size(0), m.out_dim, device=x.device, dtype=x.dtype)
        z = m.embed_latent(y0)
        for _ in range(STEPS):
            z = step_fn(m, z, xt, LAM)
        return m.decode(z)
    return solve


def wrapped(solve, d):
    def f(m, x):
        r = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return (r ** d) * solve(m, x / r)
    return f


def degree_table(m, f, x, name, d_target):
    with torch.no_grad():
        norms = [f(m, c * x).norm().item() for c in CS]
    print(f"\n  {name}   (task cần d = {d_target:+d})")
    hdr = "  " + "".join(f"{c:>11.0e}" for c in CS)
    print("    ||f(c x)||" + "".join(f"{n:>11.3e}" for n in norms))
    dh = ["   —      "]
    for i in range(1, len(CS)):
        dh.append(f"{math.log(norms[i]/max(norms[i-1],1e-300))/math.log(CS[i]/CS[i-1]):>+11.3f}")
    print("    d_hat     " + "".join(dh))
    print("    c =       " + "".join(f"{c:>11.0e}" for c in CS))
    return norms


def run(task, inp_dim, out_dim, d):
    print("\n" + "=" * 92)
    print(f"TASK: {task}   inp={inp_dim} out={out_dim}   d cần = {d:+d}")
    print("=" * 92)
    m = HopfieldEnergySolver(inp_dim, out_dim, num_heads=8, step_lr=LAM,
                             beta=None).to(DEV).to(DTYPE).eval()
    x = (torch.rand(4, inp_dim, device=DEV, dtype=DTYPE) - 0.5) * 2.0

    s_cur, s_nv, s_st = make_solver(step_current), make_solver(step_normV), make_solver(step_strict)
    degree_table(m, s_cur, x, "V1  current      (V thô)", d)
    degree_table(m, s_nv,  x, "V2  V-normalized (V qua norm_x)", d)
    degree_table(m, s_st,  x, "V3  strict-cons  (V = K W_q^T, bỏ W_v/W_o)", d)
    degree_table(m, wrapped(s_cur, d), x, "V4  wrapper      (chuẩn hóa instance + r^d)", d)


if __name__ == "__main__":
    print("\n" + "#" * 92)
    print("#  BẬC ĐỒNG NHẤT HIỆU DỤNG  d_hat  —  4 biến thể, cùng weight")
    print("#  V2/V3 kỳ vọng d_hat ~ 0 (bị chặn)   |   V4 kỳ vọng d_hat = d chính xác")
    print("#" * 92)
    run("ADDITION",       800, 400, +1)
    run("MATRIX INVERSE", 400, 400, -1)
    run("LOW-RANK",       400, 400, +1)
    print("\nĐọc: d_hat ~ 0 = output KHÔNG lớn lên khi input lớn lên = bị chặn = không ngoại suy được.")
    print("     d_hat = d ổn định ở mọi c = đồng nhất chính xác = ngoại suy đúng theo xây dựng.\n")
