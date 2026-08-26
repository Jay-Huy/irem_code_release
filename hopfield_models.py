import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Embedder:
    """Mã hóa tần số Fourier Sin/Cos cho tọa độ vị trí 1D (NeRF Positional Encoding)."""
    def __init__(self, input_dims=1, max_freq_log2=9, num_freqs=10):
        self.input_dims = input_dims
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, steps=num_freqs)
        self.out_dim = input_dims + input_dims * num_freqs * len(self.funcs)  # 1 + 1*10*2 = 21

    def embed(self, coords):
        # coords shape: (1, seq_len, 1)
        out = [coords]
        for freq in self.freq_bands:
            for fn in self.funcs:
                out.append(fn(coords * freq))
        return torch.cat(out, dim=-1)  # shape: (1, seq_len, d_pos=21)


class HopfieldEnergySolver(nn.Module):
    def __init__(self, inp_dim, out_dim, d_model=512, d_k=512, step_lr=0.5, beta=None):
        """
        Hopfield / Analytical Attention-based Energy Minimization Solver.

        Parameters:
            inp_dim:   Số phần tử đầu vào (400 cho LowRank/Inverse, 800 cho Addition)
            out_dim:   Số phần tử nghiệm đầu ra (400)
            d_model:   Tổng số chiều của Token biểu diễn (mặc định 512)
            d_k:       Số chiều không gian Query/Key (mặc định 512)
            step_lr:   Hệ số lambda cho Residual Update (mặc định 0.5)
            beta:      Nhiệt độ nghịch đảo (None -> fixed 1/sqrt(d_k), hoặc số thực -> learnable parameter)
        """
        super(HopfieldEnergySolver, self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.d_model = d_model
        self.d_k = d_k
        self.step_lr = step_lr

        # 1. Positional Embedder
        self.embedder = Embedder(input_dims=1, max_freq_log2=9, num_freqs=10)
        self.d_pos = self.embedder.out_dim        # 21
        self.d_val = self.d_model - self.d_pos    # 512 - 21 = 491

        # 2. Xử lý tham số beta
        if beta is None:
            self.learnable_beta = False
            self.beta = None
        else:
            self.learnable_beta = True
            self.beta = nn.Parameter(torch.tensor(float(beta)), requires_grad=True)

        # 3. Tạo lưới tọa độ 1D tổng quát (Đăng ký buffer để tự động chuyển thiết bị CPU/GPU)
        inp_coord = torch.Tensor(np.mgrid[:inp_dim] / inp_dim)[None, :, None]  # (1, inp_dim, 1)
        out_coord = torch.Tensor(np.mgrid[:out_dim] / out_dim)[None, :, None]  # (1, out_dim, 1)
        self.register_buffer('inp_coord', inp_coord)
        self.register_buffer('out_coord', out_coord)

        # 4. Value Projection Layers (Chiếu số thực 1D lên d_val=491)
        self.inp_linear = nn.Linear(1, self.d_val)
        self.out_linear = nn.Linear(1, self.d_val)

        # 5. Hopfield Attention Projections
        # Q: z (d_model=512) -> d_k=512
        self.W_q = nn.Linear(self.d_model, self.d_k, bias=False)
        # K: X (d_model=512) -> d_k=512
        self.W_k = nn.Linear(self.d_model, self.d_k, bias=False)
        # V: X (d_model=512) -> z (d_model=512)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)

        # 6. Khởi tạo W_v ban đầu theo đúng định lý toán học: W_v = W_k * W_q^T
        self.init_hopfield_weights()

        # 7. Layer Normalization
        self.norm_x = nn.LayerNorm(self.d_model)
        self.norm_z = nn.LayerNorm(self.d_model)

        # 8. Decode Head: Chiếu từ không gian ẩn z (d_model=512) về nghiệm số thực 1D
        self.decode_head = nn.Linear(self.d_model, 1)

    def init_hopfield_weights(self):
        """
        Khởi tạo trọng số W_v theo định lý toán học Hopfield:
        V_init = K @ W_q^T => W_v.weight = W_q.weight.T @ W_k.weight
        """
        with torch.no_grad():
            # W_q.weight: (d_k, d_model), W_k.weight: (d_k, d_model)
            # W_v.weight: (d_model, d_model)
            w_v_init = torch.matmul(self.W_q.weight.t(), self.W_k.weight)
            self.W_v.weight.copy_(w_v_init)

    def embed_input(self, x):
        """
        Nhúng dữ liệu đầu vào X thành Tokens có LayerNorm.
        Input x: Tensor shape (batch, inp_dim)
        Output:  Tensor shape (batch, inp_dim, d_model=512)
        """
        x_val = x.view(-1, self.inp_dim, 1)               # (batch, inp_dim, 1)
        x_feat = self.inp_linear(x_val)                    # (batch, inp_dim, d_val=491)
        pos_feat = self.embedder.embed(self.inp_coord)     # (1, inp_dim, d_pos=21)
        pos_feat = pos_feat.expand(x.size(0), -1, -1)      # (batch, inp_dim, d_pos=21)
        tokens = torch.cat([x_feat, pos_feat], dim=-1)     # (batch, inp_dim, d_model=512)
        return self.norm_x(tokens)

    def embed_latent(self, y):
        """
        Nhúng nghiệm ban đầu y_0 thành Latent State z_0 có LayerNorm.
        Input y: Tensor shape (batch, out_dim)
        Output:  Tensor shape (batch, out_dim, d_model=512)
        """
        y_val = y.view(-1, self.out_dim, 1)               # (batch, out_dim, 1)
        y_feat = self.out_linear(y_val)                    # (batch, out_dim, d_val=491)
        pos_feat = self.embedder.embed(self.out_coord)     # (1, out_dim, d_pos=21)
        pos_feat = pos_feat.expand(y.size(0), -1, -1)      # (batch, out_dim, d_pos=21)
        tokens = torch.cat([y_feat, pos_feat], dim=-1)     # (batch, out_dim, d_model=512)
        return self.norm_z(tokens)

    def forward_step(self, z, x_tokens, step_lr=None):
        """
        Thực hiện ĐÚNG 1 BƯỚC cập nhật trạng thái trong Latent Space:
        z_{t+1} = (1 - lambda) * z_t + lambda * Softmax(scale * Q K^T) * V

        Parameters:
            z:        Tensor shape (batch, out_dim, d_model=512) - Trạng thái ẩn hiện tại
            x_tokens: Tensor shape (batch, inp_dim, d_model=512) - Tokens đầu vào bài toán (đã qua norm_x)
            step_lr:  Hệ số lambda (nếu None thì lấy self.step_lr)
        Returns:
            z_next:   Tensor shape (batch, out_dim, d_model=512) - Trạng thái ẩn mới
        """
        lam = step_lr if step_lr is not None else self.step_lr

        Q = self.W_q(self.norm_z(z))  # (batch, out_dim, d_k=512) - Pre-LN cho Query
        K = self.W_k(x_tokens)        # (batch, inp_dim, d_k=512)
        V = self.W_v(x_tokens)        # (batch, inp_dim, d_model=512)

        # Tính scale nhiệt độ
        if self.learnable_beta:
            scale = self.beta / np.sqrt(self.d_k)
        else:
            scale = 1.0 / np.sqrt(self.d_k)

        # Tính ma trận Attention bằng torch.bmm (nhanh và tiết kiệm VRAM hơn einsum)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * scale    # (batch, out_dim, inp_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)         # (batch, out_dim, inp_dim)

        # Truy xuất thông tin từ Value Memory
        attn_out = torch.bmm(attn_weights, V)                     # (batch, out_dim, d_model=512)

        # Cập nhật trạng thái Residual thuần túy
        z_next = (1.0 - lam) * z + lam * attn_out                 # (batch, out_dim, d_model=512)
        return z_next

    def get_energy(self, z, x_tokens):
        """
        Tính hàm năng lượng Continuous Hopfield Energy chuẩn xác:
        E(z) = -(1 / scale) * logsumexp(scale * Q K^T) + 1/2 * ||z||^2
        với scale = beta / sqrt(d_k) nếu learnable_beta, ngược lại 1 / sqrt(d_k)
        Trả về: Tensor shape (batch, 1)
        """
        Q = self.W_q(self.norm_z(z))       # (batch, out_dim, d_k=512)
        K = self.W_k(x_tokens)             # (batch, inp_dim, d_k=512)

        if self.learnable_beta:
            scale = self.beta / np.sqrt(self.d_k)
        else:
            scale = 1.0 / np.sqrt(self.d_k)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * scale     # (batch, out_dim, inp_dim) = scale * Q K^T
        lse = torch.logsumexp(attn_scores, dim=-1)                # (batch, out_dim) = logsumexp(scale * Q K^T)
        quad = 0.5 * torch.sum(z ** 2, dim=-1)                   # (batch, out_dim) = 1/2 * ||z||^2

        energy_per_token = -(1.0 / scale) * lse + quad            # (batch, out_dim)
        energy = energy_per_token.mean(dim=-1, keepdim=True)      # (batch, 1)
        return energy

    def decode(self, z):
        """
        Giải mã trạng thái ẩn z ra không gian nghiệm số thực y_pred.
        Input z: Tensor shape (batch, out_dim, d_model=512)
        Output:  Tensor shape (batch, out_dim)
        """
        return self.decode_head(z).squeeze(-1)

    def forward(self, z, x_tokens, step_lr=None):
        """
        Hàm forward mặc định thực thi đúng 1 bước cập nhật Latent State.
        """
        return self.forward_step(z, x_tokens, step_lr=step_lr)
