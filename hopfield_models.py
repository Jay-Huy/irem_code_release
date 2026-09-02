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
    def __init__(self, inp_dim, out_dim, d_model=512, d_k=512, num_heads=8, step_lr=0.5,
                 beta=None, tie_mode='hard'):
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
            tie_mode:  'hard'   -> KHONG co W_v, W_o. value = K^h (W_q^h)^T, cac head CONG.
                                   Day la -grad(E1) CHINH XAC -> E giam don dieu.
                                   Nhe nhat: bo 2*d_model^2 tham so. R = 0 vinh vien.
                       'random' -> co W_v, W_o, init NGAU NHIEN (R ~ 2.0 luc init).
                                   Rang buoc THAT chi la W_v^(h) W_o^(h) = W_k^h (W_q^h)^T;
                                   tung thua so rieng le la tu do gauge GL(d_h).
                                   Ham phat tie_penalty() phai KEO tu ngoai da tap vao.
                       'orbit'  -> co W_v, W_o, init NGAY TREN da tap tai mot diem gauge
                                   NGAU NHIEN (R = 0 luc init nhung W_v != W_k).
                                   Ham phat chi phai GIU no o do.
                       'random' va 'orbit' cung function class (rong hon 'hard' vi
                       chung ROI duoc da tap); chung chi khac DUONG toi uu.
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
        assert tie_mode in ('hard', 'random', 'orbit'), f"tie_mode la {tie_mode}"
        self.tie_mode = tie_mode
        self.d_h = self.d_k // self.num_heads
        if tie_mode != 'hard':
            # V: X (d_model=512) -> z (d_model=512)
            self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
            # O: Chiếu lại không gian sau Multi-head
            self.W_o = nn.Linear(self.d_model, self.d_model, bias=False)
            if tie_mode == 'orbit':
                self.init_on_tie_manifold()
        # tie_mode == 'hard': khong tao W_v, W_o. value duoc SINH TU key.

        # 7. Layer Normalization
        self.norm_x = nn.LayerNorm(self.d_model)
        self.norm_z = nn.LayerNorm(self.d_model)

        # 8. Decode Head: Phá vỡ tổ hợp lồi bằng mạng MLP 2 lớp với SiLU
        self.decode_head = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )

    def init_on_tie_manifold(self):
        """Init W_v, W_o NGAY TREN da tap rang buoc, tai mot diem gauge NGAU NHIEN.

            W_v^(h) = W_k^h G_h ,     W_o^(h) = G_h^T (W_q^h)^T ,     G_h truc giao

        => W_v^(h) W_o^(h) = W_k^h (G_h G_h^T) (W_q^h)^T = W_k^h (W_q^h)^T   => R = 0

        Khac han viec dat W_v = W_k va W_o = W_q^T: cai do la chon G_h = I, mot diem
        TUY TIEN tren orbit 4096 chieu, va no thien lech toan bo qua trinh toi uu ve
        phia do. G_h ngau nhien thi khong.

        G_h truc giao (khong phai GL day du): G^-1 = G^T nen khong bao gio suy bien,
        va hai thua so giu duoc chuan can nhau -> conditioning cua gradient khong bi pha.
        """
        d_h, D, H = self.d_h, self.d_model, self.num_heads
        with torch.no_grad():
            for h in range(H):
                sl = slice(h * d_h, (h + 1) * d_h)
                # G_h truc giao ngau nhien qua phan tich QR
                G, _ = torch.linalg.qr(torch.randn(d_h, d_h, device=self.W_q.weight.device,
                                                   dtype=self.W_q.weight.dtype))
                # W_v.weight[sl, :] = (W_k^h G_h)^T = G_h^T @ W_k.weight[sl, :]
                self.W_v.weight[sl, :] = G.t() @ self.W_k.weight[sl, :]
                # W_o.weight[:, sl] = (G_h^T (W_q^h)^T)^T = W_q.weight[sl, :]^T @ G_h
                self.W_o.weight[:, sl] = self.W_q.weight[sl, :].t() @ G

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

        # CACH D: KHONG ap norm len z.
        # Ly do: neu Q dung norm_z(z) thi so hang -LSE chi thay HUONG cua z, con
        # 0.5*||z||^2 chi thay DO LON -> hai so hang nam trong hai khong gian con bu
        # truc giao, khong bao gio can bang nhau -> khong co diem bat dong phi tam thuong.
        # Bo norm_z thi ca hai so hang deu an z THO, dung dang Hopfield goc:
        #     E = -lse(...)  +  0.5*||z||^2      (lom + loi -> co cuc tieu, beta dieu khien basin)

        # u = self.norm_x(x_tokens)             # (batch, inp_dim, d_model) - Pre-LN, CO DINH
        u = x_tokens             # (batch, inp_dim, d_model) - Pre-LN, CO DINH

        Q = self.W_q(z)                       # (batch, out_dim, d_k=512) - z THO, khong Pre-LN
        K = self.W_k(u)                       # (batch, inp_dim, d_k=512)

        batch = Q.size(0)
        d_h = self.d_k // self.num_heads

        Qh = Q.view(batch, -1, self.num_heads, d_h).transpose(1, 2)   # (B,H,N,d_h)
        Kh = K.view(batch, -1, self.num_heads, d_h).transpose(1, 2)   # (B,H,M,d_h)

        if self.learnable_beta:
            scale = self.beta / np.sqrt(d_h)
        else:
            scale = 1.0 / np.sqrt(d_h)

        attn_weights = torch.softmax(torch.matmul(Qh, Kh.transpose(-2, -1)) * scale, dim=-1)

        if self.tie_mode == 'hard':
            # -grad(E1) CHINH XAC:  sum_h  A^h @ ( K^h (W_q^h)^T )
            # value SINH TU key, moi head cho ra du d_model chieu, cac head CONG
            # (vi E1 la TONG cac so hang LSE theo head). Khong co W_v, W_o.
            Wq_b = self.W_q.weight.view(self.num_heads, d_h, self.d_model)   # (H, d_h, D) = (W_q^h)^T
            Vh = torch.einsum('bhmk,hkd->bhmd', Kh, Wq_b)                    # (B,H,M,D)
            attn_out = torch.einsum('bhnm,bhmd->bnd', attn_weights, Vh)      # CONG theo h
        else:
            # V an u DA LN (khong phai x_tokens tho): gradient dung co value = K W_q^T,
            # ma K duoc dung tu u. Neu V an x_tokens tho thi update khong bang -grad(E1)
            # du W_v W_o co tie hoan hao.
            V = self.W_v(u)
            d_v = self.d_model // self.num_heads
            Vh = V.view(batch, -1, self.num_heads, d_v).transpose(1, 2)
            attn_out = torch.matmul(attn_weights, Vh)                        # (B,H,N,d_v)
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
            attn_out = self.W_o(attn_out)      # concat + W_o  ==  sum_h  O^h W_o^(h)

        # Cập nhật trạng thái Residual thuần túy
        z_next = (1.0 - lam) * z + lam * attn_out                 # (batch, out_dim, d_model=512)
        return z_next

    def get_energy(self, z, x_tokens): 
        """
        Tính hàm năng lượng Continuous Hopfield Energy chuẩn xác cho Multi-head:
        Tổng năng lượng của các head độc lập.
        Trả về: Tensor shape (batch, 1)
        """
        Q = self.W_q(z)                       # (batch, out_dim, d_k=512) - z THO (cach D)
        # K = self.W_k(self.norm_x(x_tokens))   # (batch, inp_dim, d_k=512)
        K = self.W_k(x_tokens)   # (batch, inp_dim, d_k=512)

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

    def tie_penalty(self, eps=1e-12):
        """R_rel = mean_h || W_v^(h) W_o^(h) - W_k^h (W_q^h)^T ||_F^2 / || W_k^h (W_q^h)^T ||_F^2

        Chi TICH bi rang buoc, khong phai tung thua so — tung thua so la tu do gauge:
            (W_v G)(G^-1 W_o) = W_v W_o   voi moi G kha nghich (d_h x d_h), 4096 chieu/head.

        Phat TOAN CUC khong du: W_v W_o = sum_h W_v^(h) W_o^(h) chi ep TONG khop, cho phep
        head nay du head kia thieu. Nhung moi head nhan voi A^h khac nhau nen tong dung
        khong cuu duoc -> phai khop TUNG head.

        CHUAN HOA theo ||target||^2 nen gia tri khong phu thuoc d_model, num_heads hay
        cach init:   ~2.0 luc init ngau nhien   |   0.0 khi tie hoan hao.
        Doc duoc nhu "phan tram lech con lai".
        """
        if self.tie_mode == 'hard':
            return torch.zeros((), device=self.W_q.weight.device)
        d_h = self.d_h
        Wq, Wk = self.W_q.weight, self.W_k.weight        # (d_k, d_model)
        Wv, Wo = self.W_v.weight, self.W_o.weight        # (d_model, d_model)
        terms = []
        for h in range(self.num_heads):
            sl = slice(h * d_h, (h + 1) * d_h)
            got = Wv.t()[:, sl] @ Wo.t()[sl, :]          # W_v^(h) W_o^(h)   (D,D)
            want = Wk.t()[:, sl] @ Wq[sl, :]             # W_k^h (W_q^h)^T   (D,D)
            terms.append((got - want).pow(2).sum() / (want.pow(2).sum() + eps))
        return torch.stack(terms).mean()

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
