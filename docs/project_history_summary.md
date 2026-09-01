# Lịch sử và Quá trình Tối ưu hóa: Hopfield Latent Reasoning

Tài liệu này tóm tắt quá trình phát triển, các vấn đề gặp phải và giải pháp tối ưu cho dự án tích hợp Modern Hopfield Network vào mô hình Algorithmic Reasoning (dựa trên codebase của IREM).

## 1. Mục tiêu và Những việc đã hoàn thành
- **Phát triển kiến trúc HopfieldEnergySolver**: Thay thế các cơ chế GNN/MLP truyền thống trong IREM/IRED bằng một mạng Attention dựa trên hàm năng lượng Hopfield liên tục. 
- **Thiết lập Pipeline Huấn luyện**: Cải tiến vòng lặp suy luận (iterative reasoning) và tích hợp deep_supervision để tính loss trên tất cả các bước unrolling.
- **Chứng minh Toán học**: Soạn thảo tài liệu nghiên cứu (Research Proposal / Discussion Note), bao gồm phụ lục chứng minh toán học chi tiết cách đạo hàm hàm năng lượng Hopfield sinh ra cơ chế Cross-Attention và Residual Connection, giúp chống lại hiện tượng suy biến hạng (Rank Collapse).

## 2. Vấn đề cốt lõi gần đây: Thất bại khi ngoại suy (OOD Extrapolation Failure)
- **Triệu chứng**: Mô hình đạt kết quả cực tốt trên tập In-Distribution (ID) nhưng thất bại thảm hại trên tập Out-of-Distribution (OOD) của bài toán Addition (Sai số MSE vọt lên ~59.61). 
- **Phân tích nguyên nhân (Root Cause)**: 
  - Vấn đề nằm ở cấu trúc **LayerNorm** trong kiến trúc Transformer tiêu chuẩn.
  - Về mặt toán học: $\text{LayerNorm}(c \cdot x) = \text{LayerNorm}(x)$.
  - Khi dữ liệu OOD to gấp  = 2.5$ lần, LayerNorm chia cho độ lệch chuẩn mới và vô tình bóp nghẹt độ lớn, biến biểu diễn của OOD thành một phân phối giống hệt tập Train. 
  - Điều này khiến mạng Attention và lớp Decode MLP mất hoàn toàn thông tin về "độ lớn thực tế" (magnitude), liên tục dự đoán ra các giá trị nhỏ thuộc dải ID, gây sai số bình phương khổng lồ với đáp án OOD.

## 3. Giải pháp Tối ưu: Chuẩn hóa tách biệt (Decoupled Norm)
Thay vì gỡ bỏ toàn bộ Norm (sẽ làm hỏng cơ chế Softmax do tích vô hướng bị scale lên ^2$ lần) hoặc giữ nguyên Norm (làm mất Scale), chúng ta đã triển khai chiến lược **Decoupled Norm**:
- **Nguyên lý kiến trúc**: 
  - *Routing (Query, Key)* cần tính bất biến (Scale-Invariant) để tính toán Attention Weights chính xác.
  - *Message Passing (Value)* cần tính đồng biến (Scale-Equivariant) để bảo toàn độ lớn của đầu ra.
- **Cập nhật Codebase (hopfield_models.py)**:
  - embed_input và embed_latent được tinh chỉnh để trả về các Token thô **chưa qua chuẩn hóa (Unnormalized)**.
  - Trong quá trình tính toán Attention (orward_step và get_energy): 
    - Truy vấn $ và Khóa $ được đi qua LayerNorm cục bộ (
orm_z và 
orm_x) để kiểm soát nhiệt độ (temperature) của Softmax.
    - Giá trị $ nhận trực tiếp tensor thô chưa chuẩn hóa (self.W_v(x_tokens)).
- **Kết quả**: Mô hình duy trì được ma trận Attention sắc nét ổn định, đồng thời mượn được thông tin độ lớn qua nhánh $ để dự đoán chính xác sự ngoại suy (Extrapolation) cho môi trường OOD.
