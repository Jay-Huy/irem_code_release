# Chỉnh cấu trúc toán của Hopfield Energy Solver

Đề xuất thay thế cho thiết kế hiện tại trong `hopfield_models.py`.

**Trạng thái:** ✅ Chương 1 · ✅ Chương 2 · tiếp theo: Chương 3

> **Ghi chú xuất xứ (2.2–2.4):** Bổ đề 1 và Bổ đề 2 là của **Energy Transformer (Hoover et al., NeurIPS 2023)**, và ET lại dẫn lại từ dòng Krotov về Lagrangian của activation. Mục 2.4 là chuyên biệt hoá ET sang cross-attention với key tĩnh. **Không có đóng góp toán học mới trong Chương 2** — đây là phần làm cho code khớp lý thuyết đã có.

> Ghi chú quyết định: đã cân nhắc và **loại** phương án "sửa energy cho khớp cross-attention" — hàm năng lượng đó không tồn tại (điều kiện khả tích: Jacobian phải đối xứng). Cũng đã **loại** phương án coi `E` là verifier học được. Cam kết hướng: **sửa `forward_step` cho khớp energy**.

---

## Chương 1 — Vấn đề: code hiện tại không khớp lý thuyết

| Mục | Nội dung | Trạng thái |
|---|---|---|
| 1.1 | Code đang tính gì — dịch `forward_step` thành công thức toán, giữ nguyên mọi LayerNorm | ✅ |
| 1.2 | Energy đang tính gì — dịch `get_energy` thành công thức toán | ✅ |
| 1.3 | Gradient **đúng** của energy ở 1.2 là gì — chain rule đầy đủ | ✅ |
| 1.4 | Đối chiếu 1.1 với 1.3 → **ba chỗ lệch**, bảng so sánh | ✅ |
| 1.5 | Hệ quả: energy không phải thế năng của động lực đang chạy ⟹ mất bảo chứng giảm, `argmin_k E` vô nghĩa | ✅ |
| 1.6 | Vấn đề thứ hai, **độc lập**: LayerNorm lột magnitude ⟹ OOD vỡ. Cơ chế: đổi thang input = đổi nhiệt độ = đổi landscape | ✅ |
| 1.7 | Vì sao ba cấu hình hiện có mỗi cái chỉ **được 2 mất 1** — và vì sao không sửa được bằng một chỗ | ✅ |

## Chương 2 — Giải pháp và chứng minh

| Mục | Nội dung | Trạng thái |
|---|---|---|
| 2.1 | Nguyên tắc: tách **hai việc** (giữ cấu trúc gradient / giữ magnitude) ra **hai chỗ** khác nhau | ✅ |
| 2.2 | **Bổ đề 1** — LayerNorm là gradient của một hàm lồi. Chứng minh từng bước, và điều kiện gain **scalar** | ✅ |
| 2.3 | **Bổ đề 2** — lấy gradient theo `g`, cộng vào `Z` thì energy vẫn giảm. Chứng minh 1 chiều → nhiều chiều → điều kiện `J ⪰ 0` | ✅ |
| 2.4 | Energy mới: viết ra, rồi lấy gradient → **ra đúng cross-attention**. Chain rule đầy đủ, đa head | ✅ |
| 2.5 | **Bổ đề 3** — hệ số thang `r^d` cho đồng nhất bậc `d` chính xác. Chứng minh + bảng theo dõi hai lần chạy + ví dụ số | ✅ |
| 2.6 | Kiểm tra tổng: cả **ba** tính chất cùng thỏa — Hopfield ✓, là gradient flow ✓, ngoại suy OOD ✓ | ✅ |

## Chương 3 — Áp dụng vào code

| Mục | Nội dung | Trạng thái |
|---|---|---|
| 3.1 | Bảng ánh xạ: mỗi công thức ở Chương 2 → dòng code nào trong `hopfield_models.py` / `train.py` | ⬜ |
| 3.2 | **Sáu thay đổi**, từng cái: code cũ → code mới → sửa lỗi nào ở Chương 1 | ⬜ |
| 3.3 | Hàm phạt `W_v W_o` — vì sao phải phạt **theo từng head**, không phải toàn cục | ⬜ |
| 3.4 | **Ba cạm bẫy**: replay buffer sai thang · gain vector của `nn.LayerNorm` · cân bằng độ lớn `ŷ_0` vs `x̂` | ⬜ |
| 3.5 | Cách xác minh: unit test đồng nhất + kỳ vọng kết quả (chưa train phải PASS) | ⬜ |
| 3.6 | Cái phải **đo** sau khi sửa: đánh đổi tham số (−44% nếu bỏ `W_v, W_o`), ID/OOD, tính đơn điệu của `E` | ⬜ |

---

## Quy ước trình bày

1. Mỗi biến đổi toán chia thành các bước đánh số, **không gộp hai bước vào một dòng**.
2. Mỗi chain rule: **viết khung trước** (liệt kê các thừa số), rồi mới điền từng thừa số.
3. Không lặp nội dung giữa các chương — Ch.1 nêu vấn đề, Ch.2 chứng minh, Ch.3 code.

## Ký hiệu dùng xuyên suốt

| Ký hiệu | Nghĩa |
|---|---|
| `M` | `inp_dim` (800 cho Addition, 400 cho LowRank/Inverse) |
| `N` | `out_dim` (400) |
| `D` | `d_model` = 512 |
| `H` | `num_heads` = 8 |
| `d_h` | `d_k / H` = 64 |
| `β` | `1/sqrt(d_h)`, hoặc `beta/sqrt(d_h)` nếu học được |
| `X_j` | token input thứ `j`, ∈ R^D |
| `Z_i` | token latent thứ `i`, ∈ R^D |
| `u_j` | `LN_x(X_j)` |
| `g_i` | `LN_z(Z_i)` |
| `J_i` | Jacobian của LayerNorm tại `Z_i`, ∈ R^{D×D} |

Quy ước vector **hàng** (khớp PyTorch).
