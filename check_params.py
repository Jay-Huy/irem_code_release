import torch
import torch.nn as nn
from models import EBM, FC, RecurrentFC, IterativeFC
from hopfield_models import HopfieldEnergySolver


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def inspect_models_for_dataset(dataset_name, inp_dim, out_dim):
    print("=" * 80)
    print(f"DATASET: {dataset_name.upper()} (inp_dim={inp_dim}, out_dim={out_dim})")
    print("=" * 80)

    # 1. EBM (IREM Baseline - Table 10)
    ebm = EBM(inp_dim, out_dim, mem=False)
    # 2. Feedforward FC Baseline
    fc = FC(inp_dim, out_dim)
    # 3. Recurrent FC (LSTM)
    rnn = RecurrentFC(inp_dim, out_dim)
    # 4. Iterative FC
    it_fc = IterativeFC(inp_dim, out_dim)
    # 5. Hopfield Energy Solver (Our Model, d_model=512)
    hopfield = HopfieldEnergySolver(inp_dim, out_dim, d_model=512, d_k=512, step_lr=0.5)

    models = {
        "IREM EBM (Table 10: 3x Linear 512)": ebm,
        "Feedforward FC (3x Linear 512)": fc,
        "Recurrent FC (LSTM)": rnn,
        "Iterative FC": it_fc,
        "Hopfield Energy Solver (Our Model)": hopfield,
    }

    for name, m in models.items():
        n_params = count_parameters(m)
        print(f"-> {name:<45}: {n_params:>10,d} parameters")

    print("\n--- ARCHITECTURE DETAIL: IREM EBM (Table 10 in Paper) ---")
    print(ebm)

    print("\n--- ARCHITECTURE DETAIL: HopfieldEnergySolver ---")
    print(hopfield)
    print("\n")


if __name__ == "__main__":
    # Task 1: Continuous Addition (inp_dim=800, out_dim=400)
    inspect_models_for_dataset("Addition", inp_dim=800, out_dim=400)

    # Task 2: Matrix Completion (inp_dim=400, out_dim=400)
    inspect_models_for_dataset("Matrix Completion (LowRank)", inp_dim=400, out_dim=400)

    # Task 3: Matrix Inverse (inp_dim=400, out_dim=400)
    inspect_models_for_dataset("Matrix Inverse", inp_dim=400, out_dim=400)
