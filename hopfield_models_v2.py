"""
HopfieldEnergySolverV2 — bản khớp với Chương 2 của docs/ToC.md

Ba tính chất được thỏa đồng thời:
  P1  beta_eff ổn định     : Q, K, V đều dựng từ token đã LayerNorm, trên input đã khử thang đo
  P2  giữ magnitude        : hệ số gauge r^d bên ngoài vòng lặp        (Bổ đề 3)
  P3  là gradient flow     : update == -lam * dÊ/dg, và Ê giảm đơn điệu (Bổ đề 1 + 2)

Tham chiếu mục:
  §2.2  LagrangianLayerNorm  — gain SCALAR dương, là gradient của hàm lồi
  §2.4  forward_step         — value sinh từ key, các head CỘNG
  §2.5  gauge r^d            — chuẩn hoá x thô TRƯỚC khi nhúng, nhân lại ở decode
  §3.3  tie_penalty          — phạt theo TỪNG head
"""

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- §2.2
class LagrangianLayerNorm(nn.Module):
    """LayerNorm với gain SCALAR gamma > 0 và bias VECTOR delta.

    Đây đúng là dL/dZ của hàm lồi
        L(Z) = D*gamma*sqrt(mean((Z - mean(Z))^2) + eps) + <delta, Z>
    nên Jacobian J = d(LN)/dZ = Hessian của L là PSD  ->  Bổ đề 2 áp dụng được.

    KHÔNG dùng nn.LayerNorm mặc định: gain per-channel (vector) làm Jacobian
    mất đối xứng -> không tồn tại L -> mất bảo chứng giảm (§2.2 phần c).
    """

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.log_gamma = nn.Parameter(torch.zeros(1))    # gamma = exp(.) > 0 luôn
        self.delta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, z):
        v = z - z.mean(dim=-1, keepdim=True)
        sigma = (v.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return self.log_gamma.exp() * v / sigma + self.delta


class FourierEmbedder(nn.Module):
    """Positional encoding Fourier log-spaced. Zero tham số. Giữ nguyên từ bản gốc."""

    def __init__(self, d_model=512, max_seq_len=400):
        super().__init__()
        freqs = torch.exp(torch.linspace(
            np.log(np.pi), np.log((max_seq_len / 2.0) * np.pi), steps=d_model // 2))
        self.register_buffer('freq_bands', freqs)

    def embed(self, coords):
        a = coords * self.freq_bands
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


# -----------------------------------------------------------------------------
class HopfieldEnergySolverV2(nn.Module):

    def __init__(self, inp_dim, out_dim, d_model=512, num_heads=8, step_lr=0.5,
                 beta=None, tie_mode='hard', soft_init='random',
                 degree=1.0, learn_degree=True):
        """
        tie_mode  : 'hard' -> KHÔNG có W_v, W_o. value = K @ W_q^T  (§2.4, ít hơn 44% tham số)
                    'soft' -> có W_v, W_o, ép bằng tie_penalty() theo TỪNG head (§3.3)
        soft_init : chỉ dùng khi tie_mode='soft'
                    'random' -> init ngẫu nhiên, để hàm phạt kéo về ràng buộc  [MẶC ĐỊNH]
                                Ràng buộc thật chỉ là W_v^(h) W_o^(h) = W_k^h (W_q^h)^T;
                                từng thừa số riêng lẻ là tự do gauge (GL(d_h), 4096 chiều/head).
                    'tied'   -> init ngay tại điểm tie. CHỈ dùng làm ĐỐI CHỨNG: nếu 'random'
                                không hội tụ mà 'tied' chạy được thì vấn đề nằm ở đường tối ưu,
                                không phải ở bản thân ràng buộc.
        degree    : bậc đồng nhất d (+1 add/lowrank, -1 inverse)
        """
        super().__init__()
        assert d_model % num_heads == 0
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.d_model, self.num_heads = d_model, num_heads
        self.d_h = d_model // num_heads
        self.step_lr = step_lr
        self.tie_mode = tie_mode

        # --- bậc đồng nhất d (§2.5) ---
        self.degree = nn.Parameter(torch.tensor(float(degree)),
                                   requires_grad=bool(learn_degree))

        # --- nhiệt độ ---
        if beta is None:
            self.learnable_beta, self.beta = False, None
        else:
            self.learnable_beta = True
            self.beta = nn.Parameter(torch.tensor(float(beta)))

        # --- toạ độ + positional ---
        self.embedder = FourierEmbedder(d_model, max(inp_dim, out_dim))
        if inp_dim == 2 * out_dim:                        # Addition: X = [R1, R2]
            c = np.mgrid[:out_dim] / float(out_dim)
            inp_c = np.concatenate([c, c], axis=0)
        else:
            inp_c = np.mgrid[:inp_dim] / float(inp_dim)
        self.register_buffer('inp_coord', torch.Tensor(inp_c)[None, :, None])
        self.register_buffer('out_coord',
                             torch.Tensor(np.mgrid[:out_dim] / float(out_dim))[None, :, None])

        # --- nhúng giá trị vô hướng -> d_model ---
        self.inp_linear = nn.Linear(1, d_model)
        self.out_linear = nn.Linear(1, d_model)

        # --- chiếu attention ---
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.soft_init = soft_init
        if tie_mode == 'soft':
            self.W_v = nn.Linear(d_model, d_model, bias=False)
            self.W_o = nn.Linear(d_model, d_model, bias=False)
            if soft_init == 'tied':                        # chỉ để đối chứng
                with torch.no_grad():
                    self.W_v.weight.copy_(self.W_k.weight)
                    self.W_o.weight.copy_(self.W_q.weight.t())
            # soft_init == 'random': giữ nguyên init mặc định của nn.Linear.
            # tie_penalty() sẽ kéo TÍCH về đúng chỗ; các thừa số tự do trôi trên orbit gauge.

        # --- LayerNorm dạng Lagrangian ---
        self.norm_x = LagrangianLayerNorm(d_model)
        self.norm_z = LagrangianLayerNorm(d_model)

        # --- decode head ---
        self.decode_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.SiLU(), nn.Linear(256, 1))

    # ------------------------------------------------------------------ §2.5
    @staticmethod
    def gauge(x):
        """r = RMS(x) trên x THÔ. Đồng nhất bậc 1. Trả (B, 1)."""
        return x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)

    def embed_input(self, x):
        """x THÔ -> (tokens của x/r, r). Mạng không bao giờ thấy thang đo."""
        r = self.gauge(x)
        xh = (x / r).view(-1, self.inp_dim, 1)
        pos = self.embedder.embed(self.inp_coord).expand(x.size(0), -1, -1)
        return self.inp_linear(xh) + pos, r

    def embed_latent(self, y_hat):
        """y_hat phải LÀ nghiệm ĐÃ chuẩn hoá (y / r^d). Xem §3.4 cạm bẫy #1."""
        yv = y_hat.view(-1, self.out_dim, 1)
        pos = self.embedder.embed(self.out_coord).expand(y_hat.size(0), -1, -1)
        return self.out_linear(yv) + pos

    # ------------------------------------------------------------------ helpers
    def _heads(self, t):
        return t.view(t.size(0), -1, self.num_heads, self.d_h).transpose(1, 2)

    def _scale(self):
        s = 1.0 / np.sqrt(self.d_h)
        return self.beta * s if self.learnable_beta else s

    def _attn(self, g, u):
        """Trả (A, K_h) — trọng số attention và key theo head."""
        Qh = self._heads(self.W_q(g))                      # (B,H,N,d_h)
        Kh = self._heads(self.W_k(u))                      # (B,H,M,d_h)
        A = torch.softmax(Qh @ Kh.transpose(-2, -1) * self._scale(), dim=-1)
        return A, Kh

    # ------------------------------------------------------------------ §2.4
    def forward_step(self, z, x_tokens, step_lr=None):
        """Û :  z <- z + lam * ( attn - g_z )

        attn = sum_h  A^h @ ( K^h W_q^{h,T} )     <- value SINH TỪ key, các head CỘNG
        """
        lam = self.step_lr if step_lr is None else step_lr
        g = self.norm_z(z)
        u = self.norm_x(x_tokens)
        A, Kh = self._attn(g, u)

        if self.tie_mode == 'hard':
            # W_q.weight: (d_model, d_model) -> khối (H, d_h, d_model) = (W_q^h)^T
            Wq_b = self.W_q.weight.view(self.num_heads, self.d_h, self.d_model)
            Vh = torch.einsum('bhmk,hkd->bhmd', Kh, Wq_b)          # (B,H,M,D)
            attn = torch.einsum('bhnm,bhmd->bnd', A, Vh)           # CỘNG theo h
        else:
            Vh = self._heads(self.W_v(u))                          # V ăn u ĐÃ LN
            o = (A @ Vh).transpose(1, 2).reshape(z.size(0), -1, self.d_model)
            attn = self.W_o(o)                                     # concat+W_o == sum_h

        return z + lam * (attn - g)

    # ------------------------------------------------------------------ §2.4 (Ê)
    def get_energy(self, z, x_tokens):
        """Ê = -1/beta * sum_h logsumexp_j( beta <Q^h_i, K^h_j> )  +  0.5*||g_i||^2

        Số hạng bậc hai trên g_i (KHÔNG phải z_i) — bắt buộc để dùng Bổ đề 2.
        """
        g = self.norm_z(z)
        u = self.norm_x(x_tokens)
        Qh, Kh = self._heads(self.W_q(g)), self._heads(self.W_k(u))
        s = self._scale()
        lse = torch.logsumexp(Qh @ Kh.transpose(-2, -1) * s, dim=-1).sum(dim=1)
        quad = 0.5 * g.pow(2).sum(dim=-1)
        return (-(1.0 / s) * lse + quad).mean(dim=-1, keepdim=True)

    # ------------------------------------------------------------------ §2.5
    def decode(self, z, r):
        """y = r^d * dec(z).  r từ embed_input."""
        return (r ** self.degree) * self.decode_head(z).squeeze(-1)

    # ------------------------------------------------------------------ §3.3
    def tie_penalty(self, eps=1e-12):
        """R_rel = mean_h  ||W_v^(h)W_o^(h) - W_k^h(W_q^h)^T||_F^2 / ||W_k^h(W_q^h)^T||_F^2

        Chỉ TÍCH bị ràng buộc, không phải từng thừa số — từng thừa số là tự do gauge:
            (W_v G)(G^-1 W_o) = W_v W_o   với mọi G khả nghịch (d_h x d_h).

        Phạt TOÀN CỤC không đủ: W_v W_o = sum_h W_v^(h) W_o^(h) chỉ ép TỔNG khớp,
        cho phép head này dư head kia thiếu. Nhưng mỗi head nhân với A^h khác nhau
        nên tổng đúng không cứu được -> phải khớp TỪNG head.

        CHUẨN HOÁ theo ||target||^2 để giá trị không phụ thuộc D, H hay cách init:
            ~2.0  lúc init ngẫu nhiên   |   0.0 khi tie hoàn hảo
        Nhờ vậy gamma nằm trong dải [0.01, 1] cho mọi cấu hình, và đọc được như
        "phần trăm lệch còn lại".
        """
        if self.tie_mode != 'soft':
            return torch.zeros((), device=self.W_q.weight.device)
        dh = self.d_h
        Wq, Wk = self.W_q.weight, self.W_k.weight          # (D, D)
        Wv, Wo = self.W_v.weight, self.W_o.weight
        terms = []
        for h in range(self.num_heads):
            sl = slice(h * dh, (h + 1) * dh)
            got = Wv.t()[:, sl] @ Wo.t()[sl, :]            # W_v^(h) W_o^(h)   (D,D)
            want = Wk.t()[:, sl] @ Wq[sl, :]               # W_k^h (W_q^h)^T   (D,D)
            terms.append((got - want).pow(2).sum() / (want.pow(2).sum() + eps))
        return torch.stack(terms).mean()

    @staticmethod
    def tie_gamma_schedule(it, gamma0=0.1, total=10000):
        """Phương pháp phạt cổ điển cho  min MSE  s.t.  R = 0: tăng gamma dần.
        Giai đoạn đầu MSE dẫn dắt hình dạng; giai đoạn sau ràng buộc siết chặt."""
        return gamma0 * (1.0 + it / max(total, 1))

    # ------------------------------------------------------------------
    def solve(self, x, num_steps, step_lr=None, y0=None):
        """Tiện ích end-to-end: x THÔ -> y THÔ. Dùng cho unit test §3.5."""
        x_tokens, r = self.embed_input(x)
        if y0 is None:
            y0 = torch.zeros(x.size(0), self.out_dim, device=x.device, dtype=x.dtype)
        z = self.embed_latent(y0)
        for _ in range(num_steps):
            z = self.forward_step(z, x_tokens, step_lr)
        return self.decode(z, r)

    def forward(self, z, x_tokens, step_lr=None):
        return self.forward_step(z, x_tokens, step_lr)
