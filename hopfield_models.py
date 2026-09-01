import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Embedder(nn.Module):
    """Mã hóa vị trí Fourier Log-spaced (Cải tiến từ NeRF, 100% Toán học, Zero parameters)"""
    def __init__(self, d_model=512, max_seq_len=400):
        super().__init__()
        num_freqs = d_model // 2  # 256 tần số cho sin và cos
        
        # Tần số thấp nhất: 1 chu kỳ toàn mảng (pi)
        # Tần số cao nhất: dao động giữa 2 điểm kề nhau (max_seq_len / 2 * pi)
        min_freq = np.log(np.pi)
        max_freq = np.log((max_seq_len / 2.0) * np.pi)
        
        # Rải đều 256 tần số trên thang log
        freq_bands = torch.exp(torch.linspace(min_freq, max_freq, steps=num_freqs))
        self.register_buffer('freq_bands', freq_bands)

    def embed(self, coords):
        # coords shape: (1, seq_len, 1)
        # angles shape: (1, seq_len, 256)
        angles = coords * self.freq_bands
        # Ghép sin và cos lại -> shape: (1, seq_len, 512)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class HopfieldEnergySolver(nn.Module):
    def __init__(self, inp_dim, out_dim, d_model=512, d_k=512, num_heads=8, step_lr=0.5, beta=None):
        """
        Hopfield / Analytical Attention-based Energy Minimization Solver.

        Parameters:
            inp_dim:   Số phần tử đầu vào (400 cho LowRank/Inverse, 800 cho Addition)
            out_dim:   Số phần tử nghiệm đầu ra (400)
            d_model:   Tổng số chiều của Token biểu diễn (mặc định 512)
            d_k:       Số chiều không gian Query/Key (mặc định 512)
            num_heads: Số lượng heads cho Multi-head Attention (mặc định 8)
            step_lr:   Hệ số lambda cho Residual Update (mặc định 0.5)
            beta:      Nhiệt độ nghịch đảo (None -> fixed 1/sqrt(d_k), hoặc số thực -> learnable parameter)
        """
        super(HopfieldEnergySolver, self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.d_model = d_model
        self.d_k = d_k
        self.num_heads = num_heads
        self.step_lr = step_lr

        # 1. Positional Embedder (Log-spaced Fourier)
        # Sử dụng đúng độ dài chuỗi tương ứng (Addition task có max_seq_len = 800)
        max_seq_len = inp_dim if inp_dim >= out_dim else out_dim
        self.embedder = Embedder(d_model=self.d_model, max_seq_len=max_seq_len)
        self.d_val = self.d_model                 # Chiếu thẳng lên 512

        # 2. Xử lý tham số beta
        if beta is None:
            self.learnable_beta = False
            self.beta = None
        else:
            self.learnable_beta = True
            self.beta = nn.Parameter(torch.tensor(float(beta)), requires_grad=True)

        # 3. Tạo lưới tọa độ 1D tổng quát (Đăng ký buffer để tự động chuyển thiết bị CPU/GPU)
        if inp_dim == 2 * out_dim:
            # Trường hợp 2 ma trận ghép nối (như bài toán Addition: X = [R1, R2])
            # Gán cùng dải tọa độ [0..out_dim-1] cho cả R1 và R2 để khớp 100% với Y
            single_coord = np.mgrid[:out_dim] / float(out_dim)
            inp_coord_np = np.concatenate([single_coord, single_coord], axis=0)  # (2*out_dim,)
        else:
            inp_coord_np = np.mgrid[:inp_dim] / float(inp_dim)

        out_coord_np = np.mgrid[:out_dim] / float(out_dim)

        inp_coord = torch.Tensor(inp_coord_np)[None, :, None]  # (1, inp_dim, 1)
        out_coord = torch.Tensor(out_coord_np)[None, :, None]  # (1, out_dim, 1)
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
        # O: Chiếu lại không gian sau Multi-head
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=False)

        # 6. Khởi tạo W_v ban đầu theo đúng định lý toán học: W_v = W_k * W_q^T
        self.init_hopfield_weights()

        # 7. Layer Normalization
        self.norm_x = nn.LayerNorm(self.d_model)
        self.norm_z = nn.LayerNorm(self.d_model)

        # 8. Decode Head: Phá vỡ tổ hợp lồi bằng mạng MLP 2 lớp với SiLU
        self.decode_head = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )

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
        Nhúng dữ liệu đầu vào X thành Tokens có LayerNorm (Sử dụng Addition thay vì Concat).
        Input x: Tensor shape (batch, inp_dim)
        Output:  Tensor shape (batch, inp_dim, d_model=512)
        """
        x_val = x.view(-1, self.inp_dim, 1)               # (batch, inp_dim, 1)
        x_feat = self.inp_linear(x_val)                   # (batch, inp_dim, d_model=512)
        pos_feat = self.embedder.embed(self.inp_coord)    # (1, inp_dim, d_model=512)
        pos_feat = pos_feat.expand(x.size(0), -1, -1)     # (batch, inp_dim, 512)
        
        # Cộng thay vì nối (Addition instead of Concatenation)
        tokens = x_feat + pos_feat                        # (batch, inp_dim, 512)
        return tokens

    def embed_latent(self, y):
        """
        Nhúng nghiệm ban đầu y_0 thành Latent State z_0 có LayerNorm (Sử dụng Addition thay vì Concat).
        Input y: Tensor shape (batch, out_dim)
        Output:  Tensor shape (batch, out_dim, d_model=512)
        """
        y_val = y.view(-1, self.out_dim, 1)               # (batch, out_dim, 1)
        y_feat = self.out_linear(y_val)                   # (batch, out_dim, d_model=512)
        pos_feat = self.embedder.embed(self.out_coord)    # (1, out_dim, d_model=512)
        pos_feat = pos_feat.expand(y.size(0), -1, -1)     # (batch, out_dim, 512)
        
        # Cộng thay vì nối (Addition instead of Concatenation)
        tokens = y_feat + pos_feat                        # (batch, out_dim, 512)
        return tokens

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

        Q = self.W_q(self.norm_z(z))          # (batch, out_dim, d_k=512) - Pre-LN cho Query
        K = self.W_k(self.norm_x(x_tokens))   # (batch, inp_dim, d_k=512) - Pre-LN cho Key

        V = self.W_v(x_tokens)                # (batch, inp_dim, d_model=512) - Không Norm cho Value
        # V = self.W_v(self.norm_x(x_tokens))   # (batch, inp_dim, d_model=512)

        batch = Q.size(0)
        d_k_head = self.d_k // self.num_heads
        d_v_head = self.d_model // self.num_heads

        # Chia thành nhiều heads: (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch, -1, self.num_heads, d_k_head).transpose(1, 2)
        K = K.view(batch, -1, self.num_heads, d_k_head).transpose(1, 2)
        V = V.view(batch, -1, self.num_heads, d_v_head).transpose(1, 2)

        # Tính scale nhiệt độ cho từng head
        if self.learnable_beta:
            scale = self.beta / np.sqrt(d_k_head)
        else:
            scale = 1.0 / np.sqrt(d_k_head)

        # Tính ma trận Attention độc lập cho từng Head
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (batch, num_heads, out_dim, inp_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Truy xuất thông tin từ Value Memory độc lập cho từng Head
        attn_out = torch.matmul(attn_weights, V)                    # (batch, num_heads, out_dim, d_v_head)

        # Ghép (Concat) kết quả của các Heads lại
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        
        # Chiếu lại không gian (Output Projection)
        attn_out = self.W_o(attn_out)

        # Cập nhật trạng thái Residual thuần túy
        z_next = (1.0 - lam) * z + lam * attn_out                 # (batch, out_dim, d_model=512)
        return z_next

    def get_energy(self, z, x_tokens): 
        """
        Tính hàm năng lượng Continuous Hopfield Energy chuẩn xác cho Multi-head:
        Tổng năng lượng của các head độc lập.
        Trả về: Tensor shape (batch, 1)
        """
        Q = self.W_q(self.norm_z(z))          # (batch, out_dim, d_k=512)
        K = self.W_k(self.norm_x(x_tokens))   # (batch, inp_dim, d_k=512)

        batch = Q.size(0)
        d_k_head = self.d_k // self.num_heads

        # Chia thành nhiều heads
        Q = Q.view(batch, -1, self.num_heads, d_k_head).transpose(1, 2)
        K = K.view(batch, -1, self.num_heads, d_k_head).transpose(1, 2)

        if self.learnable_beta:
            scale = self.beta / np.sqrt(d_k_head)
        else:
            scale = 1.0 / np.sqrt(d_k_head)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale     # (batch, num_heads, out_dim, inp_dim)
        
        # LogSumExp cho từng head
        lse = torch.logsumexp(attn_scores, dim=-1)                # (batch, num_heads, out_dim)
        
        # Tổng năng lượng của tất cả các heads
        lse = lse.sum(dim=1)                                      # (batch, out_dim)
        
        # Thành phần bình phương (giữ nguyên trên toàn bộ d_model)
        quad = 0.5 * torch.sum(z ** 2, dim=-1)                    # (batch, out_dim)

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
