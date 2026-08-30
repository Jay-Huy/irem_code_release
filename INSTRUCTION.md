# HƯỚNG DẪN THỰC THI & KHẢO SÁT THAM SỐ (INSTRUCTION MANUAL)
Tài liệu này cung cấp:
1. **Đối chiếu chi tiết giữa Báo cáo trong Paper (Table 1, 2, 10) và Codebase**.
2. **Khảo sát cấu trúc mạng & số lượng tham số (Parameters Comparison)**.
3. **Cơ chế Logging, Tần suất Test & Giải thích các Metrics (`train` & `test`)**.
4. **Từ điển toàn bộ các biến/cờ CLI trong hệ thống (W&B bật mặc định & tùy chỉnh `run_name`)**.
5. **Các câu lệnh mẫu sẵn sàng chạy**.

---
# I. ĐỐI CHIẾU THAM SỐ: PAPER VS CODEBASE GỐC VS HOPFIELD
Dưới đây là bảng so sánh chéo từng siêu tham số (Hyperparameters) giữa báo cáo trong Paper và mặc định của repository:
| Tham số / Setting | Báo cáo trong Paper | Mặc định Codebase gốc | Khuyến nghị (Recommended) | Ghi chú & Lý giải |
| :--- | :--- | :--- | :--- | :--- |
| **Batch Size** | **128** | `512` | `--batch_size 128` (hoặc `512`) | Paper báo cáo dùng `batch_size = 128` trên 1 GPU Titan X. Codebase để mặc định `512`. Nếu dùng GPU có VRAM lớn $\ge 8\text{GB}$, cả 128 và 512 đều chạy rất tốt. |
| **Optimizer** | **Adam** | Adam | Adam | Khớp 100%. |
| **Learning Rate (LR)** | **1e-4** | `1e-4` | `--lr 1e-4` | Khớp 100%. |
| **Số bước huấn luyện (Iterations)** | **10,000** | 10,000 (Mặc định) | `--num_iterations 10000` (hoặc tùy chọn 1000, 5000) | Có thể tùy chỉnh số bước huấn luyện bất kỳ qua cờ `--num_iterations`. |
| **Training Iterative Steps ($K$)** | **5 steps** | `10` steps | `--num_steps 5` (hoặc `10`) | Paper báo cáo train với 5 bước lặp (`num_steps = 5`), code để mặc định `10`. Với Hopfield Solver, train 5 hay 10 bước đều hội tụ rất nhanh. |
| **Inference Iterative Steps (Test)** | **80 steps** | 80 steps | 80 steps | Trong hàm `test()`, hệ thống tự động chạy suy luận 80 bước để đo lường khả năng thích ứng (Adaptive Compute). |
| **Step Size $\lambda$ (`step_lr`)** | Không cố định | `100.0` (EBM) | **`0.5`** (Hopfield)<br>`100.0` (EBM) | EBM gốc dùng `100.0` vì gradient MLP rất bé. Hopfield là Residual Convex combination nên $\lambda \in (0, 1]$ (Khuyến nghị: `0.5`, có thể thử nghiệm `0.25`, `0.75`, `1.0`). |
| **Replay Buffer** | Có dùng (Capacity 10k) | Bật mặc định | Bật (mặc định) hoặc `--no_replay_buffer` | EBM bắt buộc cần Buffer để không bị kẹt attractor ảo. Hopfield có thể so sánh đối chứng giữa Có vs Không có Buffer. |
| **Tần suất Log Loss (`--log_interval`)** | — | `10` steps | `--log_interval 10` (hoặc `1`) | Cứ mỗi 10 steps sẽ in ra màn hình terminal và đẩy loss lên W&B / TensorBoard. |
| **Tần suất Test & Save (`--save_interval`)** | — | `1000` steps | `--save_interval 1000` (hoặc `200`) | Cứ mỗi 1000 steps sẽ chạy đánh giá test 80 bước trên 1000 mẫu và lưu checkpoint `model_latest.pth`. |
| **Tracking / Logging** | Weights & Biases | TensorBoard | **W&B (Bật mặc định)** | Đã tích hợp W&B tự động, hỗ trợ tùy chỉnh `--run_name`. |
---

# II. KHẢO SÁT KIẾN TRÚC MÔ HÌNH & SỐ LƯỢNG THAM SỐ (MODEL PARAMETERS)
Dựa trên **Table 10 trong Appendix D của Paper**:
* Mạng IREM EBM được cấu tạo bởi:
  $$\text{Linear } 512 \rightarrow \text{Swish} \rightarrow \text{Linear } 512 \rightarrow \text{Swish} \rightarrow \text{Linear } 512 \rightarrow \text{Swish} \rightarrow \text{Linear } 1$$
* Đầu vào của mạng là vector ghép nối $(\mathbf{x}, \mathbf{y})$:
  * Với bài toán `Addition`: $\mathbf{x} \in \mathbb{R}^{800}, \mathbf{y} \in \mathbb{R}^{400} \implies \text{Input dim} = 1200$.
  * Với bài toán `LowRank` & `Inverse`: $\mathbf{x} \in \mathbb{R}^{400}, \mathbf{y} \in \mathbb{R}^{400} \implies \text{Input dim} = 800$.
### Bảng so sánh số lượng tham số (Param Count):
| Mô hình | Task Addition (inp=800, out=400) | Task LowRank / Inverse (inp=400, out=400) | Kiến trúc chi tiết |
| :--- | :--- | :--- | :--- |
| **IREM EBM (Table 10)** | **1,140,737** params (~1.14M) | **935,937** params (~0.94M) | MLP 3 lớp ẩn 512 + Swish $\rightarrow$ Scalar Output (1) |
| **Feedforward FC** | 1,140,737 params | 935,937 params | MLP 3 lớp ẩn 512 + ReLU $\rightarrow$ Vector Output (400) |
| **Recurrent FC (LSTM)** | 234,817 params | 234,817 params | Linear(inp, 196) + LSTM(196, 196) + Linear(196, out) |
| **HopfieldEnergySolver ($d=512$)** | **790,957** params (~0.79M) | **790,957** params (~0.79M) | Embedder (21) + Linear(1, 491) + LayerNorm + Attention ($W_q, W_k, W_v \in \mathbb{R}^{512 \times 512}$) + Decode |
> [!NOTE]
> Để tự động in chi tiết kiến trúc và đo tham số, bạn có thể chạy:
> ```bash
> python check_params.py
> ```
---

# III. CƠ CHẾ LOGGING, TẦN SUẤT TEST & GIẢI THÍCH METRICS
### 1. Tần suất Log Loss khi Huấn luyện (`--log_interval`, mặc định: `10` steps)
* **Vị trí thiết lập**: `train.py` dòng 155 (`--log_interval 10`) và dòng 591–635 trong hàm `train()`.
* **Cơ chế hoạt động**: Cứ mỗi `log_interval` bước (ví dụ: step 0, 10, 20, 30...):
  1. In ra màn hình Terminal: `Iteration {it} im_loss: {:.6f}  mean_last_dist: {:.6f}`.
  2. Đẩy các metrics sau lên Weights & Biases (W&B):
     * `train/im_loss`: MSE Loss trên batch hiện tại ở bước suy luận cuối cùng $K$.
     * `train/loss`: Tổng loss sau khi cộng regularizer (nếu có).
     * `train/mean_last_dist`: Sai số trung bình giữa prediction và ground truth.
     * `train/no_replay_loss` & `train/replay_loss`: Loss phân tách giữa mẫu mới và mẫu lấy từ Replay Buffer.
     * `train/energy_no_replay` & `train/energy_replay`: Năng lượng Hopfield tại bước cuối.
     * `train/iteration`: Số bước lặp huấn luyện hiện tại.
### 2. Tần suất Đánh giá Test & Lưu Checkpoint (`--save_interval`, mặc định: `1000` steps)
* **Vị trí thiết lập**: `train.py` dòng 157 (`--save_interval 1000`) và dòng 637–647 trong hàm `train()`.
* **Cơ chế hoạt động**: Cứ mỗi `save_interval` bước (step 0, 1000, 2000, 3000...):
  1. Lưu file checkpoint: `cachedir/<exp>/model_latest.pth`.
  2. Kích hoạt hàm `test()` thực hiện **80 bước suy luận liên tục** trên 1000 bài toán kiểm thử độc lập.
  3. Đẩy các metrics chuẩn của bài báo lên W&B:
     * `test/error_step_10`: Sai số MSE sau 10 bước suy luận.
     * `test/error_step_20`: Sai số MSE sau 20 bước suy luận.
     * `test/error_step_40`: Sai số MSE sau 40 bước suy luận.
     * `test/error_step_80`: Sai số MSE sau 80 bước suy luận.
     * `test/min_energy_error`: Sai số tại điểm có năng lượng Hopfield nhỏ nhất $\arg\min_k E(\mathbf{z}_k)$.
     * `test/best_error`: **Sai số nhỏ nhất toàn thời gian (Metric chính đối chiếu với Table 1 & Table 2 trong Paper)**.
---

# IV. TỪ ĐIỂN THAM SỐ DÒNG LỆNH (CLI ARGUMENTS DICTIONARY)
### 1. Chọn Mô hình (Model Architecture)
* `--hopfield`: **[KHUYÊN DÙNG]** Kích hoạt mô hình `HopfieldEnergySolver` (Analytical Attention Gradient + LayerNorm + $d_{\text{model}}=512$).
* (Không truyền gì): Mặc định chạy mô hình `EBM` của IREM (MLP 3 lớp Swish 512 + PyTorch Autograd).
* `--recurrent`: Baseline RNN (LSTM).
* `--iterative_decoder`: Baseline Feedforward lặp tự hồi quy.
* `--ponder`: Baseline PonderNet.
* `--decoder`: Baseline MLP 1 bước duy nhất.
### 2. Thiết lập Huấn luyện & Dữ liệu
* `--dataset`: Tên tập dữ liệu (`addition`, `lowrank`, `inverse`).
* `--train`: Bật chế độ huấn luyện (nếu không truyền sẽ chỉ chạy test).
* `--cuda`: Chạy trên GPU CUDA.
* `--batch_size`: Kích thước batch (Mặc định paper: `128`, mặc định code: `512`).
* `--lr`: Learning rate cho Adam optimizer (Mặc định: `1e-4`).
* `--num_iterations`: **[MỚI]** Tổng số bước huấn luyện Iterations/Batches cần chạy (Mặc định: `10000`, có thể chỉnh ví dụ `1000`, `5000` tùy ý).
* `--num_steps`: Số bước suy luận lặp lúc Train (Mặc định paper: `5`, mặc định code: `10`).
* `--step_lr`: Hệ số bước nhảy $\lambda$ cập nhật Residual cho Hopfield (Mặc định: `0.5`).
* `--beta`: Tham số nhiệt độ nghịch đảo (Mặc định: `None` $\rightarrow \frac{1}{\sqrt{d_k}}$; nếu truyền số thực như `1.0` $\rightarrow$ trở thành `nn.Parameter` tự học).
* `--log_interval`: Tần suất log train loss ra terminal và W&B (Mặc định: `10` steps).
* `--save_interval`: Tần suất chạy test 80 steps và lưu checkpoint (Mặc định: `1000` steps).
* `--no_replay_buffer`: **Tắt Replay Buffer** (Huấn luyện độc lập 100% với mẫu ngẫu nhiên mới).
* `--truncate_hopfield`: Bật ngắt gradient ở các bước đầu (Mặc định: `False` $\rightarrow$ Full Backprop).
* `--ood`: Đánh giá trên tập dữ liệu ngoại suy khó hơn (Out-of-Distribution).
* `--exp`: Tên thí nghiệm lưu file cục bộ (lưu trong `cachedir/` và `result/`).

### 3. Theo dõi qua Weights & Biases (`wandb`)
* *(Mặc định bật tự động)*: Nếu chưa cài đặt `wandb`, hệ thống sẽ báo lỗi yêu cầu chạy `pip install wandb`.
* `--run_name`: **[MỚI]** Tùy chỉnh tên hiển thị của Run trên W&B Dashboard (ví dụ: `--run_name hopfield_addition_run1`).
* `--no_wandb`: Tắt ghi log W&B (nếu chỉ muốn chạy offline).
* `--wandb_project`: Tên project trên W&B (Mặc định: `irem-experiments`).
* `--wandb_entity`: Tên team / username trên W&B (Tùy chọn).
---

# V. MẪU CELL CHẠY TRÊN KAGGLE / GOOGLE COLAB (TỰ ĐỘNG HÓA THAM SỐ)
Trong Notebook trên Kaggle hoặc Colab, bạn chỉ cần gán biến một lần ở đầu, toàn bộ các cell huấn luyện và đánh giá sẽ tự động đồng bộ theo:

### 1. Script 1: Mô hình Baseline (1 head, không Deep Supervision)
```python
# ==========================================
# 1. KHAI BÁO CÁC BIẾN THỰC NGHIỆM
# ==========================================
DATASET = "lowrank"          # "addition", "lowrank", "inverse"
NUM_ITERS = 1000   # 500, 1000, 10000
NUM_STEPS = 2
SAVE_INTERVAL = 200

STEP_LR = 0.5                # 0.25, 0.5, 0.75, 1.0
BATCH_SIZE = 128

# 2. Tự động sinh tên Experiment và W&B Run Name đồng bộ
EXP_NAME = f"hopfield_{DATASET}_{NUM_ITERS}iter_{NUM_STEPS}step_{STEP_LR}lr_baseline"
RUN_NAME = f"hopfield_{DATASET}_{NUM_ITERS}iter_{NUM_STEPS}step_{STEP_LR}lr_baseline"

# Chạy huấn luyện (Baseline)
!python train.py --dataset {DATASET} --train --cuda \
    --batch_size {BATCH_SIZE} --num_steps {NUM_STEPS} --step_lr {STEP_LR} \
    --hopfield --num_heads 1 --save_interval {SAVE_INTERVAL} \
    --num_iterations {NUM_ITERS} \
    --exp {EXP_NAME} --run_name {RUN_NAME}
```

```python
# Đánh giá cột "Same Diff." (In-Distribution)
!python train.py --dataset {DATASET} --cuda \
    --resume_iter {NUM_ITERS} \
    --hopfield --num_heads 1 --exp {EXP_NAME} \
    --run_name {EXP_NAME}_eval_same
```

```python
# Đánh giá cột "Harder Diff." (Out-of-Distribution / --ood)
!python train.py --dataset {DATASET} --cuda \
    --resume_iter {NUM_ITERS} \
    --hopfield --num_heads 1 --exp {EXP_NAME} \
    --ood \
    --run_name {EXP_NAME}_eval_harder
```

### 2. Script 2: Mô hình Best (8 heads, có Deep Supervision)
```python
# ==========================================
# 1. KHAI BÁO CÁC BIẾN THỰC NGHIỆM
# ==========================================
DATASET = "lowrank"          # "addition", "lowrank", "inverse"
NUM_ITERS = 1000   # 500, 1000, 10000
NUM_STEPS = 2
SAVE_INTERVAL = 200

STEP_LR = 0.5                # 0.25, 0.5, 0.75, 1.0
BATCH_SIZE = 128

# 2. Tự động sinh tên Experiment và W&B Run Name đồng bộ
EXP_NAME = f"hopfield_{DATASET}_{NUM_ITERS}iter_{NUM_STEPS}step_{STEP_LR}lr_best"
RUN_NAME = f"hopfield_{DATASET}_{NUM_ITERS}iter_{NUM_STEPS}step_{STEP_LR}lr_best"

# Chạy huấn luyện (Best)
!python train.py --dataset {DATASET} --train --cuda \
    --batch_size {BATCH_SIZE} --num_steps {NUM_STEPS} --step_lr {STEP_LR} \
    --hopfield --num_heads 8 --deep_sup --save_interval {SAVE_INTERVAL} \
    --num_iterations {NUM_ITERS} \
    --exp {EXP_NAME} --run_name {RUN_NAME}
```

```python
# Đánh giá cột "Same Diff." (In-Distribution)
!python train.py --dataset {DATASET} --cuda \
    --resume_iter {NUM_ITERS} \
    --hopfield --num_heads 8 --exp {EXP_NAME} \
    --run_name {EXP_NAME}_eval_same
```

```python
# Đánh giá cột "Harder Diff." (Out-of-Distribution / --ood)
!python train.py --dataset {DATASET} --cuda \
    --resume_iter {NUM_ITERS} \
    --hopfield --num_heads 8 --exp {EXP_NAME} \
    --ood \
    --run_name {EXP_NAME}_eval_harder
```