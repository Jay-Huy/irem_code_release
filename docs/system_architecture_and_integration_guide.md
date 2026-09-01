# HỆ THỐNG IREM CODEBASE: HƯỚNG DẪN THỰC NGHIỆM HOPFIELD ENERGY SOLVER

Tài liệu này ghi nhận toàn bộ các tham số mới, cơ chế tích hợp và các câu lệnh mẫu để bạn tiến hành thực nghiệm so sánh sòng phẳng giữa mô hình **IREM gốc (EBM + Autograd)** và mô hình mới **Hopfield Energy Solver (Analytical Attention Gradient)**.

---

# PHẦN 1: CÁC THAM SỐ CLI MỚI (ARGUMENTS & FLAGS)

Trong `train.py`, hệ thống đã được tích hợp đầy đủ các cờ điều khiển:

```
========================================================================================================================
CỜ LỆNH (FLAG)              KIỂU DỮ LIỆU    MẶC ĐỊNH        MÔ TẢ CHI TIẾT
========================================================================================================================
--hopfield                  boolean         False           Bật mô hình HopfieldEnergySolver (Analytical Gradient).
--step_lr                   float           100.0 (EBM)     Hệ số lambda cập nhật Residual: (1 - lam)*z + lam*Attn.
                                            -> 0.5 (Hopfield) Tự động điều chỉnh về 0.5 cho Hopfield nếu > 1.0.
--beta                      float           None            Nhiệt độ nghịch đảo (None -> 1/sqrt(d_k), hoặc số thực learnable).
--truncate_hopfield         boolean         False           Ngắt gradient ở các bước đầu (Mặc định: False -> Full Backprop).
--no_replay_buffer          boolean         False           Tắt Replay Buffer (Dùng 100% noise mới cho mỗi batch).
--dataset                   string          negate          Tên tập dữ liệu (addition, lowrank, inverse).
--num_steps                 int             10              Số bước lặp suy luận lúc Train.
--exp                       string          default         Tên thư mục log thực nghiệm trong cachedir/ và result/.
========================================================================================================================
```

---

# PHẦN 2: THIẾT KẾ CÁC THỰC NGHIỆM SO SÁNH (ABLATION EXPERIMENTS)

### Thực nghiệm 1: So sánh Hiệu năng Cốt lõi (EBM gốc vs Hopfield Solver)
Đánh giá sai số MSE sau 80 bước suy luận trên 3 tập dữ liệu.

* **Task 1: Continuous Addition**
  * *EBM gốc*:
    ```bash
    python train.py --dataset addition --train --cuda --batch_size 512 --num_steps 10 --step_lr 100.0 --exp ebm_addition
    ```
  * *Hopfield Solver*:
    ```bash
    python train.py --dataset addition --train --cuda --batch_size 512 --num_steps 10 --step_lr 0.5 --hopfield --exp hopfield_addition
    ```

* **Task 2: Matrix Completion (`LowRankDataset`)**
  * *EBM gốc*:
    ```bash
    python train.py --dataset lowrank --train --cuda --batch_size 512 --num_steps 10 --step_lr 100.0 --exp ebm_lowrank
    ```
  * *Hopfield Solver*:
    ```bash
    python train.py --dataset lowrank --train --cuda --batch_size 512 --num_steps 10 --step_lr 0.5 --hopfield --exp hopfield_lowrank
    ```

* **Task 3: Matrix Inverse**
  * *EBM gốc*:
    ```bash
    python train.py --dataset inverse --train --cuda --batch_size 512 --num_steps 10 --step_lr 100.0 --exp ebm_inverse
    ```
  * *Hopfield Solver*:
    ```bash
    python train.py --dataset inverse --train --cuda --batch_size 512 --num_steps 10 --step_lr 0.5 --hopfield --exp hopfield_inverse
    ```

---

### Thực nghiệm 2: Đánh giá Vai trò của Replay Buffer (Có vs Không Replay Buffer)
Nhờ tính chất hội tụ đơn điệu của Mạng Hopfield trên các attractor, ta kiểm chứng giả thuyết: *Hopfield không cần Replay Buffer vẫn hội tụ tốt và tránh được attractor ảo*.

* **Hopfield CÓ Replay Buffer**:
  ```bash
  python train.py --dataset addition --train --cuda --batch_size 512 --hopfield --step_lr 0.5 --exp hopfield_with_replay
  ```
* **Hopfield KHÔNG CÓ Replay Buffer (`--no_replay_buffer`)**:
  ```bash
  python train.py --dataset addition --train --cuda --batch_size 512 --hopfield --step_lr 0.5 --no_replay_buffer --exp hopfield_no_replay
  ```

---

### Thực nghiệm 3: Khảo sát Hệ số Bước nhảy $\lambda$ (`--step_lr`)
Kiểm tra tốc độ hội tụ với $\lambda \in \{0.25, 0.5, 0.75, 1.0\}$:
* $\lambda = 0.25$: `--step_lr 0.25` (Trượt dốc êm, ổn định cao).
* $\lambda = 0.50$: `--step_lr 0.5` (Cân bằng mặc định).
* $\lambda = 0.75$: `--step_lr 0.75` (Tiến nhanh về cực tiểu).
* $\lambda = 1.00$: `--step_lr 1.0` (Fixed-point pure update: $z_{new} = \text{Attn}$).

---

### Thực nghiệm 4: Khảo sát Cơ chế Gradient (Full Backprop vs Truncated)
* **Full Backprop (Mặc định cho Hopfield)**:
  Lan truyền ngược qua toàn bộ chuỗi $N$ bước lặp.
* **Truncated Backprop (`--truncate_hopfield`)**:
  Ngắt computational graph ở các bước trung gian, chỉ tính gradient ở bước cuối cùng:
  ```bash
  python train.py --dataset addition --train --cuda --batch_size 512 --hopfield --truncate_hopfield --exp hopfield_truncated
  ```

---

# PHẦN 3: LOGGING VÀ THEO DÕI KẾT QUẢ

1. **Xem tiến độ trên Terminal**:
   Cứ sau mỗi 1000 iteration (`--save_interval 1000`), hàm `test()` sẽ tự động chạy suy luận $80$ bước và in ra các chỉ số:
   ```
   best test error (10, 20, 40, 80, min_energy): ...
   ```
2. **Xem đồ thị TensorBoard**:
   ```bash
   tensorboard --logdir cachedir/
   ```
