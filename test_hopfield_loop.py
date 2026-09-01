import torch
from hopfield_models_v2 import HopfieldEnergySolverV2 as M
for mode in ['hard', 'soft']:
    m = M(800, 400, num_heads=8, tie_mode=mode, degree=1.0).double().eval()
    x = (torch.rand(2, 800, dtype=torch.float64) - 0.5) * 2
    c = 7.3
    y1, y2 = m.solve(x, 10), m.solve(c * x, 10)
    d = ((y2 - c * y1).norm() / (c * y1).norm()).item()
    print(mode, 'params', sum(p.numel() for p in m.parameters()), 'homog_dev %.2e' % d, 'tie_R %.3f' % m.tie_penalty().item())