"""
KIỂM ĐỊNH TÍNH ĐỒNG NHẤT (HOMOGENEITY) CỦA HOPFIELD SOLVER
==========================================================

Mục đích: kiểm tra bằng ĐẠI SỐ (không cần train) xem model có thỏa
          model(c*x) == c^d * model(x)   với mọi c > 0
          d = +1 cho addition / lowrank,  d = -1 cho inverse.

Test này KHÔNG kiểm tra độ chính xác. Một model weight random sẽ PASS
nếu cấu trúc đúng, và vẫn dự đoán sai bét. Đó chính là mục đích:
tách "cấu trúc có đúng không" khỏi "model có học được không".

Chạy:  python test_homogeneity.py
"""

# pyrefly: ignore [missing-import]
import torch
import numpy as np
from hopfield_models_v2 import HopfieldEnergySolverV2

torch.manual_seed(0)
np.random.seed(0)

DEV   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64          # float64 để loại nhiễu số học khỏi kết luận
STEPS = 10                     # num_steps như lúc train
LAM   = 0.5                    # step_lr


# ----------------------------------------------------------------------
# Vòng lặp suy luận, sao y gen_answer() trong train.py (nhánh --hopfield)
# ----------------------------------------------------------------------
def solve_raw(model, x, num_steps=STEPS, lam=LAM):
    """Model HIỆN TẠI: không có wrapper. z_0 = 0 để test tiền định."""
    x_tokens = model.embed_input(x)
    y0 = torch.zeros(x.size(0), model.out_dim, device=x.device, dtype=x.dtype)
    z = model.embed_latent(y0)
    for _ in range(num_steps):
        z = model.forward_step(z, x_tokens, step_lr=lam)
    return model.decode(z)


def solve_wrapped(model, x, d, num_steps=STEPS, lam=LAM):
    """Model CÓ WRAPPER: chuẩn hóa cả instance -> giải -> nhân lại r^d."""
    r = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)     # (B,1) một scalar / instance
    y = solve_raw(model, x / r, num_steps, lam)
    return (r ** d) * y


# ----------------------------------------------------------------------
def check_homogeneity(fn, x, c, d, label):
    """Trả về (pass, độ lệch tương đối)."""
    with torch.no_grad():
        y1 = fn(x)
        y2 = fn(c * x)
    target = (c ** d) * y1
    dev = (y2 - target).norm() / target.norm().clamp_min(1e-30)
    ok  = torch.allclose(y2, target, rtol=1e-6, atol=1e-10)
    flag = "PASS ✅" if ok else "FAIL ❌"
    print(f"    {label:<34s} {flag}   lệch tương đối = {dev.item():.6e}")
    return ok, dev.item()


def saturation_sweep(fn_raw, x, d, label):
    """Bằng chứng thực nghiệm cho ĐỊNH LÝ BỊ CHẶN:
       ||f(c x)|| phải phẳng (bị chặn), trong khi c^d ||f(x)|| tăng."""
    print(f"\n    {label}")
    print(f"    {'c':>10s} | {'||f(c·x)|| ĐO ĐƯỢC':>22s} | {'c^d·||f(x)|| PHẢI RA':>22s} | {'tỉ lệ':>10s}")
    print(f"    {'-'*10}-+-{'-'*22}-+-{'-'*22}-+-{'-'*10}")
    with torch.no_grad():
        base = fn_raw(x).norm().item()
        for c in [1.0, 2.5, 10.0, 100.0, 1e3, 1e4]:
            got  = fn_raw(c * x).norm().item()
            want = (c ** d) * base
            print(f"    {c:>10.1f} | {got:>22.6f} | {want:>22.6f} | {got/max(want,1e-30):>10.2e}")


# ----------------------------------------------------------------------
def run_task(name, inp_dim, out_dim, d, num_heads=8):
    print("\n" + "=" * 84)
    print(f"TASK: {name}   (inp_dim={inp_dim}, out_dim={out_dim}, bậc đồng nhất d={d:+d})")
    print("=" * 84)

    model = HopfieldEnergySolverV2(inp_dim, out_dim, num_heads=num_heads,
                                 step_lr=LAM, beta=None).to(DEV).to(DTYPE)
    model.eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  Tham số: {n_par:,}   |   device: {DEV}   |   dtype: {DTYPE}")

    # input mô phỏng phân phối train: U(-1, 1)
    x = (torch.rand(4, inp_dim, device=DEV, dtype=DTYPE) - 0.5) * 2.0
    c = 7.3

    print(f"\n  [A] KIỂM ĐỊNH ĐỒNG NHẤT với c = {c}")
    ok_raw, dev_raw = check_homogeneity(lambda t: solve_raw(model, t),
                                       x, c, d, "Model HIỆN TẠI (không wrapper)")
    ok_wrp, dev_wrp = check_homogeneity(lambda t: solve_wrapped(model, t, d),
                                        x, c, d, "Model CÓ WRAPPER")

    print(f"\n  [B] BẰNG CHỨNG CHO ĐỊNH LÝ BỊ CHẶN (model hiện tại)")
    saturation_sweep(lambda t: solve_raw(model, t), x, d,
                     "Nếu trạng thái bị chặn -> cột ĐO ĐƯỢC phẳng, cột PHẢI RA tăng, tỉ lệ -> 0")

    return dict(task=name, d=d, ok_raw=ok_raw, dev_raw=dev_raw,
                ok_wrp=ok_wrp, dev_wrp=dev_wrp)


if __name__ == "__main__":
    print("\n" + "#" * 84)
    print("#  KIỂM ĐỊNH ĐỒNG NHẤT — HopfieldEnergySolverV2")
    print("#  Kỳ vọng:  model hiện tại FAIL   |   model có wrapper PASS")
    print("#" * 84)

    results = []
    results.append(run_task("ADDITION",      inp_dim=800, out_dim=400, d=+1))
    results.append(run_task("MATRIX INVERSE", inp_dim=400, out_dim=400, d=-1))
    results.append(run_task("LOW-RANK",      inp_dim=400, out_dim=400, d=+1))

    print("\n" + "=" * 84)
    print("TỔNG KẾT")
    print("=" * 84)
    print(f"{'Task':<18s} {'d':>3s} | {'hiện tại':>10s} {'lệch':>13s} | {'có wrapper':>12s} {'lệch':>13s}")
    print("-" * 84)
    for r in results:
        print(f"{r['task']:<18s} {r['d']:>+3d} | "
              f"{'PASS' if r['ok_raw'] else 'FAIL':>10s} {r['dev_raw']:>13.3e} | "
              f"{'PASS' if r['ok_wrp'] else 'FAIL':>12s} {r['dev_wrp']:>13.3e}")
    print("-" * 84)
    print("\nĐọc kết quả:")
    print("  * 'hiện tại FAIL'  -> xác nhận model chưa có đối xứng scale.")
    print("  * 'có wrapper PASS' với weight RANDOM -> đối xứng là tính chất CẤU TRÚC,")
    print("    không phải thứ học được. Đây là điều cần chứng minh.")
    print("  * Bảng [B]: cột ĐO ĐƯỢC phẳng khi c tăng = trạng thái bị chặn = định lý đúng.\n")
