import torch
from hopfield_models import HopfieldEnergySolver

def test_hopfield_energy_flow():
    batch_size = 4
    inp_dim = 800
    out_dim = 400
    model = HopfieldEnergySolver(inp_dim, out_dim, d_model=512, d_k=512, step_lr=0.5)

    inp = torch.randn(batch_size, inp_dim)
    pred = torch.randn(batch_size, out_dim)

    # 1. Embeddings
    x_tokens = model.embed_input(inp)
    z = model.embed_latent(pred)

    assert x_tokens.shape == (batch_size, inp_dim, 512)
    assert z.shape == (batch_size, out_dim, 512)

    # 2. Energy
    energy = model.get_energy(z, x_tokens)
    assert energy.shape == (batch_size, 1), f"Expected (4, 1), got {energy.shape}"

    # 3. 80-step loop simulation
    energies = []
    preds = [pred]
    for i in range(80):
        z = model.forward_step(z, x_tokens, step_lr=0.5)
        p = model.decode(z)
        e = model.get_energy(z, x_tokens)
        preds.append(p)
        energies.append(e)

    energies_tensor = torch.stack(energies, dim=0)
    assert energies_tensor.shape == (80, batch_size, 1), f"Expected (80, 4, 1), got {energies_tensor.shape}"

    # Test argmin energy indexing
    min_idx = energies_tensor[:, :, 0].argmin(dim=0)[None, :]
    assert min_idx.shape == (1, batch_size), f"Expected (1, 4), got {min_idx.shape}"

    print("ALL TESTS PASSED SUCCESSFULLY! Energy shape and min_idx verified 100%!")

if __name__ == "__main__":
    test_hopfield_energy_flow()
