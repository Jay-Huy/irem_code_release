import os
os.environ["WANDB_START_METHOD"] = "thread"
import torch
from models import EBM, FC, IterativeFC, IterativeAttention, IterativeFCAttention, \
    IterativeTransformer, EBMTwin, RecurrentFC, PonderFC
from hopfield_models import HopfieldEnergySolver
import torch.nn.functional as F
import pdb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
from dataset import LowRankDataset, ShortestPath, Negate, Inverse, Square, Identity, \
    Det, LU, Sort, Eigen, QR, Equation, FiniteWrapper, Parity, Addition
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os.path as osp
import numpy as np
from imageio import imwrite
import argparse
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
from torchvision.utils import make_grid
import seaborn as sns


def worker_init_fn(worker_id):
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, inputs):
        batch_size = len(inputs)
        if self._next_idx >= len(self._storage):
            self._storage.extend(inputs)
        else:
            if batch_size + self._next_idx < self._maxsize:
                self._storage[self._next_idx:self._next_idx +
                              batch_size] = inputs
            else:
                split_idx = self._maxsize - self._next_idx
                self._storage[self._next_idx:] = inputs[:split_idx]
                self._storage[:batch_size - split_idx] = inputs[split_idx:]
        self._next_idx = (self._next_idx + batch_size) % self._maxsize

    def _encode_sample(self, idxes):
        inps = []
        opts = []
        targets = []
        scratchs = []

        # Store in the intermediate state of optimization problem
        for i in idxes:
            inp, opt, target, scratch = self._storage[i]
            opt = opt
            inps.append(inp)
            opts.append(opt)
            targets.append(target)
            scratchs.append(scratch)

        inps = np.array(inps)
        opts = np.array(opts)
        targets = np.array(targets)
        scratchs = np.array(scratchs)

        return inps, opts, targets, scratchs

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes), torch.Tensor(idxes)

    def set_elms(self, data, idxes):
        if len(self._storage) < self._maxsize:
            self.add(data)
        else:
            for i, ix in enumerate(idxes):
                self._storage[ix] = data[i]


"""Parse input arguments"""
parser = argparse.ArgumentParser(description='Train EBM model')

parser.add_argument('--train', action='store_true',
                    help='whether or not to train')
parser.add_argument('--cuda', action='store_true',
                    help='whether to use cuda or not')
parser.add_argument('--no_replay_buffer', action='store_true',
                    help='do not use a replay buffer to train models')
parser.add_argument('--dataset', default='negate', type=str,
                    help='dataset to evaluate')
parser.add_argument('--logdir', default='cachedir', type=str,
                    help='location where log of experiments will be stored')
parser.add_argument('--exp', default='default', type=str,
                    help='name of experiments')
parser.add_argument('--run_name', default=None, type=str,
                    help='custom run name for Weights & Biases (defaults to exp name)')
parser.add_argument('--no_wandb', action='store_true',
                    help='disable logging to Weights & Biases (W&B is enabled by default)')
parser.add_argument('--wandb_project', default='irem-experiments', type=str,
                    help='Weights & Biases project name')
parser.add_argument('--wandb_entity', default=None, type=str,
                    help='Weights & Biases entity name')

# training
parser.add_argument('--resume_iter', default=0, type=int,
                    help='iteration to resume training')
parser.add_argument('--batch_size', default=512, type=int,
                    help='size of batch of input to use')
parser.add_argument('--num_epoch', default=10000, type=int,
                    help='number of epochs of training to run')
parser.add_argument('--num_iterations', default=10000, type=int,
                    help='total number of training iterations (batches) to run (defaults to 10000)')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='learning rate for training')
parser.add_argument('--log_interval', default=10, type=int,
                    help='log outputs every so many batches')
parser.add_argument('--save_interval', default=1000, type=int,
                    help='save outputs every so many batches')

default_workers = 0 if os.name == 'nt' else 4
parser.add_argument('--data_workers', default=default_workers, type=int,
                    help='Number of different data workers to load data in parallel')

# Model specific settings
parser.add_argument('--rank', default=20, type=int,
                    help='rank of matrix to use')
parser.add_argument('--num_steps', default=10, type=int,
                    help='Steps of gradient descent for training')
parser.add_argument('--test_steps', default=20, type=int,
                    help='Steps of iterative inference for testing')
parser.add_argument('--step_lr', default=100.0, type=float,
                    help='step size of latents')
parser.add_argument('--recurrent', action='store_true',
                    help='utilize a recurrent model to output prediction')
parser.add_argument('--ponder', action='store_true',
                    help='utilize a ponder network model to output prediction')
parser.add_argument('--decoder', action='store_true',
                    help='utilize a decoder network to output prediction')
parser.add_argument('--iterative_decoder', action='store_true',
                    help='utilize a decoder to output prediction')
parser.add_argument('--hopfield', action='store_true',
                    help='utilize hopfield energy solver model to output prediction')
parser.add_argument('--beta', default=None, type=float,
                    help='inverse temperature beta for Hopfield attention (default: None -> 1/sqrt(d_k))')
parser.add_argument('--num_heads', default=8, type=int,
                    help='Number of heads for Hopfield Multi-head Attention (default: 8. Set to 1 for Single-head)')
parser.add_argument('--tie_mode', default='hard', type=str,
                    choices=['hard', 'random', 'orbit'],
                    help='how W_v W_o is tied to W_k W_q^T. '
                         'hard: no W_v/W_o at all (update is exactly -grad E). '
                         'random: free W_v/W_o, random init, tie enforced only by tie_penalty. '
                         'orbit: free W_v/W_o initialised on the tie manifold (random orthogonal gauge G_h).')
parser.add_argument('--tie_gamma', default=0.01, type=float,
                    help='weight of tie_penalty() in the loss. Default 0.01 chosen so that '
                         '||grad R|| ~ 10%% of ||grad MSE|| at init (see check_descent.py PHAN 3). '
                         'Ignored when tie_mode=hard (penalty is identically 0).')
parser.add_argument('--deep_sup', action='store_true',
                    help='Enable Deep Supervision (calculate loss across all iterative steps)')
parser.add_argument('--truncate_hopfield', action='store_true',
                    help='truncate gradient in Hopfield reasoning loop (default: False -> full backprop)')
parser.add_argument('--mem', action='store_true',
                    help='add external memory to compute answers')
parser.add_argument('--no_truncate', action='store_true',
                    help='don"t truncate gradient backprop')

# Distributed training hyperparameters
parser.add_argument('--gpus', default=1, type=int,
                    help='number of gpus to train with')
parser.add_argument('--node_rank', default=0, type=int, help='rank of node')
parser.add_argument('--capacity', default=50000, type=int,
                    help='number of elements to generate')
parser.add_argument('--infinite', action='store_true',
                    help='makes the dataset have an infinite number of elements')


best_energy_error = float('inf')
best_oracle_error = float('inf')


def average_gradients(model):
    size = float(dist.get_world_size())

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def gen_answer(inp, FLAGS, model, pred, scratchpad, num_steps, create_graph=True):
    """
        Implement iterative reasoning to obtain the answer to a problem
    """

    # List of intermediate predictions
    preds = []
    im_grads = []
    energies = []
    logits = []

    if FLAGS.decoder:
        pred = model.forward(inp)
        preds = [pred]
        im_grad = torch.zeros(1)
        im_grads = [im_grad]
        energies = [torch.zeros(1)]
    elif FLAGS.recurrent:
        preds = []
        im_grad = torch.zeros(1)
        im_grads = [im_grad]
        energies = [torch.zeros(1)]
        state = None
        for i in range(num_steps):
            pred, state = model.forward(inp, state)
            preds.append(pred)
    elif FLAGS.ponder:
        im_merge = torch.cat([pred, inp], dim=-1)

        preds, logits = model.forward(im_merge, iters=num_steps)
        pred = preds[-1]
        im_grad = torch.zeros(1)
        im_grads = [im_grad]
        energies = [torch.zeros(1)]
        state = None

    elif FLAGS.iterative_decoder:
        for i in range(num_steps):
            energy = torch.zeros(1)

            noise_add = (torch.rand_like(pred) - 0.5)
            out_dim = model.out_dim

            im_merge = torch.cat([pred, inp], dim=-1)
            pred = model.forward(im_merge) + pred

            preds.append(pred)
            energies.append(torch.zeros(1))
            im_grads.append(torch.zeros(1))
    elif FLAGS.hopfield:
        # Precompute input tokens once
        x_tokens = model.embed_input(inp)
        # Embed initial guess / replay sample into latent state z_0
        z = model.embed_latent(pred)

        preds = [pred]
        im_grads = [torch.zeros(1, device=inp.device)]
        energies = []

        step_lr = FLAGS.step_lr if FLAGS.step_lr <= 1.0 else 0.5

        for i in range(num_steps):
            # Only truncate if explicitly requested by flag and not on the last step
            if getattr(FLAGS, 'truncate_hopfield', False) and (i != num_steps - 1):
                z = z.detach()

            z = model.forward_step(z, x_tokens, step_lr=step_lr)
            pred_step = model.decode(z)
            with torch.no_grad():
                energy = model.get_energy(z, x_tokens)

            preds.append(pred_step)
            energies.append(energy)
            im_grads.append(torch.zeros(1, device=inp.device))

        pred = preds[-1]
    else:
        with torch.enable_grad():
            pred.requires_grad_(requires_grad=True)
            s = inp.size()
            scratchpad.requires_grad_(requires_grad=True)
            preds.append(pred)

            for i in range(num_steps):
                if FLAGS.mem:
                    im_merge = torch.cat([pred, inp, scratchpad], dim=-1)
                else:
                    im_merge = torch.cat([pred, inp], dim=-1)

                energy = model.forward(im_merge)

                if FLAGS.mem:
                    im_grad, scratchpad_grad = torch.autograd.grad(
                        [energy.sum()], [pred, scratchpad], create_graph=create_graph)
                else:

                    if FLAGS.no_truncate:
                        im_grad, = torch.autograd.grad(
                            [energy.sum()], [pred], create_graph=create_graph)
                    else:
                        if i != (num_steps - 1):
                            im_grad, = torch.autograd.grad(
                                [energy.sum()], [pred], create_graph=False)
                        else:
                            im_grad, = torch.autograd.grad(
                                [energy.sum()], [pred], create_graph=create_graph)

                pred = pred - FLAGS.step_lr * im_grad

                if FLAGS.mem:
                    scratchpad = scratchpad - FLAGS.step_lr * scratchpad
                    scratchpad = torch.clamp(scratchpad, -1, 1)

                preds.append(pred)
                energies.append(energy)
                im_grads.append(im_grad)

    return pred, preds, im_grads, energies, scratchpad, logits


def ema_model(model, model_ema, mu=0.999):
    for (model, model_ema) in zip(model, model_ema):
        for param, param_ema in zip(
                model.parameters(), model_ema.parameters()):
            param_ema.data[:] = mu * param_ema.data + (1 - mu) * param.data


def sync_model(model):
    size = float(dist.get_world_size())

    for param in model.parameters():
        dist.broadcast(param.data, 0)


def init_model(FLAGS, device, dataset):
    if FLAGS.decoder:
        model = FC(dataset.inp_dim, dataset.out_dim)
    elif FLAGS.recurrent:
        model = RecurrentFC(dataset.inp_dim, dataset.out_dim)
    elif FLAGS.ponder:
        model = PonderFC(dataset.inp_dim, dataset.out_dim, FLAGS.num_steps)
    elif FLAGS.iterative_decoder:
        model = IterativeFC(dataset.inp_dim, dataset.out_dim, FLAGS.mem)
    elif FLAGS.hopfield:
        step_lr = FLAGS.step_lr if FLAGS.step_lr <= 1.0 else 0.5
        beta = getattr(FLAGS, 'beta', None)
        num_heads = getattr(FLAGS, 'num_heads', 8)
        tie_mode = getattr(FLAGS, 'tie_mode', 'hard')
        model = HopfieldEnergySolver(
            dataset.inp_dim, dataset.out_dim,
            num_heads=num_heads, step_lr=step_lr, beta=beta,
            tie_mode=tie_mode
        )
    else:
        model = EBM(dataset.inp_dim, dataset.out_dim, FLAGS.mem)

    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)

    return model, optimizer


def safe_cumprod(t, eps=1e-10, dim=-1):
    t = torch.clip(t, min=eps, max=1.)
    return torch.exp(torch.cumsum(torch.log(t), dim=dim))


def exclusive_cumprod(t, dim=-1):
    cum_prod = safe_cumprod(t, dim=dim)
    return pad_to(cum_prod, (1, -1), value=1., dim=dim)


def calc_geometric(l, dim=-1):
    return exclusive_cumprod(1 - l, dim=dim) * l


def _eval_loader(loader, model, FLAGS, dev, test_steps, tag, step):
    """Chay test_steps buoc suy dien tren <=11 batch dau cua `loader`.

    Tra ve dict:
        dist    : (test_steps+1,)  MSE trung binh tai tung buoc (index 0 = y_0 ngau nhien)
        energy  : (test_steps,)    gia tri E trung binh tai tung buoc
        oracle_err / oracle_step  : min MSE thuc su (dung ground-truth de chon buoc)
        energy_err / energy_step  : MSE tai buoc co E nho nhat (cach chon cua EBM)
        mono_ok / mono_tot        : so cap buoc lien tiep co E giam / tong so cap
    """
    from tqdm import tqdm

    dist_list, energy_list = [], []
    min_energy_dist_list, min_energy_step_list = [], []
    oracle_dist_list, oracle_step_list = [], []
    mono_ok, mono_tot = 0, 0
    counter = 0

    with torch.no_grad():
        for inp, im in tqdm(loader, desc=f"test[{tag}] @ it{step}", leave=False):
            im = im.float().to(dev)
            inp = inp.float().to(dev)

            # Initialize prediction from random guess
            pred = (torch.rand_like(im) - 0.5) * 2
            scratch = torch.zeros_like(inp)

            pred, preds, im_grad, energies, scratch, logits = gen_answer(
                inp, FLAGS, model, pred, scratch, test_steps)

            preds = torch.stack(preds, dim=0)        # (test_steps + 1, batch, out_dim)
            energies = torch.stack(energies, dim=0)  # (test_steps, batch, 1)

            dist = (preds - im[None, :])
            dist = torch.pow(dist, 2).mean(dim=-1)   # (test_steps + 1, batch)
            dist_energies = dist[1:, :]              # (test_steps, batch)

            # --- Energy Best (EBM selection) ---
            min_idx = energies[:, :, 0].argmin(dim=0)[None, :]
            dist_min_energy = torch.gather(dist_energies, 0, min_idx)
            min_energy_dist_list.append(dist_min_energy.detach().squeeze(0))
            min_energy_step_list.append(min_idx.detach().squeeze(0))

            # --- Oracle Best (Absolute Minimum MSE) ---
            oracle_min_dist, oracle_min_idx = dist_energies.min(dim=0)
            oracle_dist_list.append(oracle_min_dist.detach())
            oracle_step_list.append(oracle_min_idx.detach())

            # --- E don dieu: dem so cap buoc lien tiep ma E KHONG tang ---
            e = energies[:, :, 0]                    # (test_steps, batch)
            if e.size(0) > 1:
                inc = (e[1:] > e[:-1] + 1e-9)
                mono_ok += int((~inc).sum().item())
                mono_tot += int(inc.numel())

            dist_list.append(dist.mean(dim=-1).detach())
            energy_list.append(energies.mean(dim=-1).mean(dim=-1).detach())

            counter += 1
            if counter > 10:   # Evaluate on a subset of batches to speed up testing
                break

    return {
        'dist': torch.stack(dist_list, dim=0).mean(dim=0),
        'energy': torch.stack(energy_list, dim=0).mean(dim=0),
        'oracle_err': torch.cat(oracle_dist_list).mean().item(),
        'oracle_step': torch.cat(oracle_step_list).float().mean().item() + 1,
        'energy_err': torch.cat(min_energy_dist_list).mean().item(),
        'energy_step': torch.cat(min_energy_step_list).float().mean().item() + 1,
        'mono_ok': mono_ok,
        'mono_tot': mono_tot,
    }


def test(test_loader_id, test_loader_ood, model, FLAGS, step=0, is_best=False):
    """Danh gia CA HAI phan phoi (in-distribution va OOD) va in mot bang gon.

    Tra ve oracle_best_error cua IN-DISTRIBUTION -> day la tieu chi chon model_best.
    """
    global best_energy_error, best_oracle_error

    dev = torch.device("cuda") if FLAGS.cuda else torch.device("cpu")
    model.eval()

    test_steps = getattr(FLAGS, 'test_steps', 20)

    res = {'ID': _eval_loader(test_loader_id, model, FLAGS, dev, test_steps, 'ID', step)}
    if test_loader_ood is not None:
        res['OOD'] = _eval_loader(test_loader_ood, model, FLAGS, dev, test_steps, 'OOD', step)

    best_energy_error = min(best_energy_error, res['ID']['energy_err'])
    best_oracle_error = min(best_oracle_error, res['ID']['oracle_err'])

    # ------------------------------------------------------------------ in bang
    cols = sorted(set(m for m in [test_steps // 4, test_steps // 2,
                                  3 * test_steps // 4, test_steps] if m >= 1))
    print()
    rule = f"── test @ iter {step} "
    print(rule + "─" * max(4, 12 + 11 * len(cols) + 34 - len(rule)))
    hdr = " " * 12 + "".join(f"{'step'+str(m):>11s}" for m in cols)
    hdr += f"{'oracle(step)':>17s}{'energy(step)':>17s}"
    print(hdr)
    for tag in ['ID', 'OOD']:
        if tag not in res:
            continue
        r = res[tag]
        row = f"  {tag:<10s}"
        for m in cols:
            row += f"{r['dist'][m].item():>11.3e}" if m < r['dist'].numel() else f"{'-':>11s}"
        row += f"{r['oracle_err']:>11.3e}({r['oracle_step']:>4.1f})"
        row += f"{r['energy_err']:>11.3e}({r['energy_step']:>4.1f})"
        print(row)

    if FLAGS.hopfield:
        mono = "  E don dieu:"
        for tag in ['ID', 'OOD']:
            if tag in res and res[tag]['mono_tot'] > 0:
                r = res[tag]
                mono += f"    {tag} {r['mono_ok']}/{r['mono_tot']}"
        print()
        print(mono)

    tail = "  ← MOI TOT NHAT, da luu model_best.pth" if is_best else ""
    print(f"  best(ID oracle): {best_oracle_error:.3e}{tail}")
    print()

    # ------------------------------------------------------------------ wandb
    if getattr(FLAGS, 'use_wandb', False) and HAS_WANDB:
        try:
            log_dict = {
                "test/global_best_oracle": best_oracle_error,
                "test/global_best_energy": best_energy_error,
            }
            for tag in res:
                p = 'test' if tag == 'ID' else 'test_ood'
                r = res[tag]
                log_dict[f"{p}/oracle_best_error"] = r['oracle_err']
                log_dict[f"{p}/oracle_best_step"] = r['oracle_step']
                log_dict[f"{p}/energy_best_error"] = r['energy_err']
                log_dict[f"{p}/energy_best_step"] = r['energy_step']
                if r['mono_tot'] > 0:
                    log_dict[f"{p}/E_mono_frac"] = r['mono_ok'] / r['mono_tot']
                for m in cols:
                    if m < r['dist'].numel():
                        log_dict[f"{p}/error_step_{m}"] = r['dist'][m].item()
            wandb.log(log_dict, step=step)
        except Exception:
            pass

    model.train()
    return res['ID']['oracle_err']


def train(train_dataloader, test_loader_id, test_loader_ood, logger, model,
          optimizer, FLAGS, logdir, rank_idx):

    it = FLAGS.resume_iter
    optimizer.zero_grad()
    dev = torch.device("cuda")

    # initalize a replay buffer of solutions
    replay_buffer = ReplayBuffer(10000)

    for epoch in range(FLAGS.num_epoch):
        for inp, im in train_dataloader:
            im = im.float().to(dev)
            inp = inp.float().to(dev)

            # Initalize a solution from random
            pred = (torch.rand_like(im) - 0.5) * 2

            # Sample a proportion of samples from past optimization results
            if FLAGS.replay_buffer and len(replay_buffer) >= FLAGS.batch_size:
                replay_batch, _ = replay_buffer.sample(im.size(0))
                inp_replay, opt_replay, gt_replay, scratch_replay = replay_batch

                replay_mask = np.concatenate( [np.ones(im.size(0)), np.zeros(im.size(0))]).astype(bool)
                inp = torch.cat([torch.Tensor(inp_replay).cuda(), inp], dim=0)
                pred = torch.cat([torch.Tensor(opt_replay).cuda(), pred], dim=0)
                im = torch.cat([torch.Tensor(gt_replay).cuda(), im], dim=0)
            else:
                replay_mask = (
                    np.random.uniform(
                        0,
                        1,
                        im.size(0)) > 1.0)

            scratch = torch.zeros_like(inp)

            num_steps = FLAGS.num_steps
            pred, preds, im_grads, energies, scratch, logits = gen_answer(
                inp, FLAGS, model, pred, scratch, num_steps)
            energies = torch.stack(energies, dim=0)
            preds = torch.stack(preds, dim=1)

            im_grads = torch.stack(im_grads, dim=1)

            if FLAGS.ponder:
                geometric_dist = calc_geometric(torch.full(
                    (FLAGS.num_steps,), 1 / FLAGS.num_steps, device=dev))
                halting_probs = calc_geometric(logits.sigmoid(), dim=1)[..., 0]

            if FLAGS.decoder:
                im_loss = torch.pow(
                    preds[:, -1:] - im[:, None, :], 2).mean(dim=-1).mean(dim=-1)
            elif FLAGS.ponder:
                halting_probs = halting_probs / \
                    halting_probs.sum(dim=1)[:, None]
                im_loss = (torch.pow(
                    preds[:, :] - im[:, None, :], 2)).mean(dim=-1).mean(dim=-1)
            elif FLAGS.hopfield:
                if getattr(FLAGS, 'deep_sup', False):
                    # preds[:, 0] la y_0 ngau nhien (chua qua buoc nao), khong phai
                    # output cua model -> KHONG dua vao loss.
                    im_loss = torch.pow(preds[:, 1:] - im[:, None, :], 2).mean(dim=-1).mean(dim=-1)
                else:
                    im_loss = torch.pow(preds[:, -1:] - im[:, None, :], 2).mean(dim=-1).mean(dim=-1)
            else:
                im_loss = torch.pow(
                    preds[:, -1:] - im[:, None, :], 2).mean(dim=-1).mean(dim=-1)

            loss = im_loss.mean()

            # im_loss_last: MSE cua RIENG buoc cuoi. Luon tinh, ke ca khi deep_sup bat,
            # de con so nay so sanh duoc giua cac run co/khong deep_sup.
            with torch.no_grad():
                im_loss_last = torch.pow(
                    preds[:, -1] - im, 2).mean().item()

            tie_R = 0.0
            if FLAGS.hopfield:
                # tie_mode='hard' -> tie_penalty() tra ve 0 (khong co W_v/W_o),
                # nen khong can re nhanh o day.
                R = model.tie_penalty()
                tie_R = R.item()
                if FLAGS.tie_gamma != 0.0:
                    loss = loss + FLAGS.tie_gamma * R

            if FLAGS.ponder:
                ponder_loss = 0.01 * \
                    F.kl_div(torch.log(geometric_dist[None, :] + 1e-10), halting_probs, None, None, 'batchmean')
                loss = loss + ponder_loss

            loss.backward()

            if FLAGS.replay_buffer:
                inp_replay = inp.cpu().detach().numpy()
                pred_replay = pred.cpu().detach().numpy()
                im_replay = im.cpu().detach().numpy()
                scratch = scratch.cpu().detach().numpy()
                encode_tuple = list(zip(list(inp_replay), list(
                    pred_replay), list(im_replay), list(scratch)))

                replay_buffer.add(encode_tuple)

            if FLAGS.gpus > 1:
                average_gradients(model)

            optimizer.step()
            optimizer.zero_grad()

            if it % FLAGS.log_interval == 0 and rank_idx == 0:
                loss = loss.item()
                kvs = {}
                kvs['im_loss'] = im_loss.mean().item()
                kvs['im_loss_last'] = im_loss_last

                if FLAGS.hopfield:
                    kvs['tie_R'] = tie_R

                if it > 10:
                    replay_mask = replay_mask
                    no_replay_mask = ~replay_mask
                    kvs['no_replay_loss'] = im_loss[no_replay_mask].mean().item()
                    kvs['replay_loss'] = im_loss[replay_mask].mean().item()

                    if FLAGS.ponder:
                        kvs['ponder_loss'] = ponder_loss

                    if (not FLAGS.iterative_decoder) and (not FLAGS.decoder) and (
                            not FLAGS.recurrent) and (not FLAGS.ponder):
                        kvs['energy_no_replay'] = energies[-1,
                                                           no_replay_mask].mean().item()
                        kvs['energy_replay'] = energies[-1,
                                                        replay_mask].mean().item()

                        kvs['energy_start_no_replay'] = energies[0,
                                                                 no_replay_mask].mean().item()
                        kvs['energy_start_replay'] = energies[0,
                                                              replay_mask].mean().item()

                mean_last_dist = torch.abs(pred - im).mean()
                kvs['mean_last_dist'] = mean_last_dist.item()

                string = "Iteration {} ".format(it)

                for k, v in kvs.items():
                    string += "%s: %.6f  " % (k, v)
                    logger.add_scalar(k, v, it)

                if getattr(FLAGS, 'use_wandb', False) and rank_idx == 0 and HAS_WANDB:
                    try:
                        wandb_dict = {f"train/{k}": v for k, v in kvs.items()}
                        wandb_dict["train/loss"] = loss
                        wandb_dict["train/iteration"] = it
                        wandb.log(wandb_dict, step=it)
                    except Exception as e:
                        pass

                print(string)

            if it % FLAGS.save_interval == 0 and rank_idx == 0:
                ckpt = {'FLAGS': FLAGS, 'iter': it}
                ckpt['model_state_dict'] = model.state_dict()
                ckpt['optimizer_state_dict'] = optimizer.state_dict()

                # model_latest: LUON ghi -> dung de resume.
                torch.save(ckpt, osp.join(logdir, "model_latest.pth"))

                # Danh gia CHINH weight vua ghi (khong load lai file nao).
                prev_best = best_oracle_error
                cur = test(test_loader_id, test_loader_ood, model, FLAGS,
                           step=it, is_best=False)

                # model_best: chi ghi khi oracle_best_error tren ID tot hon truoc do.
                if cur < prev_best - 1e-12:
                    ckpt['best_oracle_error'] = cur
                    torch.save(ckpt, osp.join(logdir, "model_best.pth"))
                    print(f"  → model_best.pth updated: oracle {prev_best:.3e} → {cur:.3e}")

            if it >= getattr(FLAGS, 'num_iterations', 10000):
                print(f"\n[INFO] Đã hoàn thành huấn luyện: {it}/{FLAGS.num_iterations} iterations.")
                return

            it += 1


def main_single(rank, FLAGS):
    global best_oracle_error, best_energy_error
    rank_idx = rank
    world_size = FLAGS.gpus
    logdir = osp.join(FLAGS.logdir, FLAGS.exp)

    if not os.path.exists('result/%s' % FLAGS.exp):
        try:
            os.makedirs('result/%s' % FLAGS.exp)
        except BaseException:
            pass

    if not os.path.exists(logdir):
        try:
            os.makedirs('logdir')
        except BaseException:
            pass

    # Load Dataset
    # test_dataset      = IN-DISTRIBUTION  (cung phan phoi voi train)
    # test_dataset_ood  = OOD              (None neu task khong co bien the OOD)
    test_dataset_ood = None
    if FLAGS.dataset == 'lowrank':
        dataset = LowRankDataset('train', FLAGS.rank, False)
        test_dataset = LowRankDataset('test', FLAGS.rank, False)
        test_dataset_ood = LowRankDataset('test', FLAGS.rank, True)
    elif FLAGS.dataset == 'shortestpath':
        dataset = ShortestPath('train', FLAGS.rank, FLAGS.num_steps)
        test_dataset = ShortestPath('test', FLAGS.rank, FLAGS.num_steps)
    elif FLAGS.dataset == 'negate':
        dataset = Negate('train', FLAGS.rank)
        test_dataset = Negate('test', FLAGS.rank)
    elif FLAGS.dataset == 'addition':
        dataset = Addition('train', FLAGS.rank, False)
        test_dataset = Addition('test', FLAGS.rank, False)
        test_dataset_ood = Addition('test', FLAGS.rank, True)
    elif FLAGS.dataset == 'inverse':
        dataset = Inverse('train', FLAGS.rank, False)
        test_dataset = Inverse('test', FLAGS.rank, False)
        test_dataset_ood = Inverse('test', FLAGS.rank, True)
    elif FLAGS.dataset == 'square':
        dataset = Square('train', FLAGS.rank, FLAGS.num_steps)
        test_dataset = Square('test', FLAGS.rank, FLAGS.num_steps)
    elif FLAGS.dataset == 'identity':
        dataset = Identity('train', FLAGS.rank, FLAGS.num_steps)
        test_dataset = Identity('test', FLAGS.rank, FLAGS.num_steps)
    elif FLAGS.dataset == 'det':
        dataset = Det('train', FLAGS.rank)
        test_dataset = Det('test', FLAGS.rank)
    elif FLAGS.dataset == 'lu':
        dataset = LU('train', FLAGS.rank)
        test_dataset = LU('test', FLAGS.rank)
    elif FLAGS.dataset == 'sort':
        dataset = Sort('train', FLAGS.rank, FLAGS.num_steps)
        test_dataset = Sort('test', FLAGS.rank, FLAGS.num_steps)
    elif FLAGS.dataset == 'eigen':
        dataset = Eigen('train', FLAGS.rank, FLAGS.num_steps)
        test_dataset = Eigen('test', FLAGS.rank, FLAGS.num_steps)
    elif FLAGS.dataset == 'equation':
        dataset = Equation('train', FLAGS.rank, FLAGS.num_steps)
        test_dataset = Equation('test', FLAGS.rank, FLAGS.num_steps)
    elif FLAGS.dataset == 'qr':
        dataset = QR('train', FLAGS.rank, FLAGS.num_steps)
        test_dataset = QR('test', FLAGS.rank, FLAGS.num_steps)
    elif FLAGS.dataset == 'parity':
        dataset = Parity('train', FLAGS.rank, FLAGS.num_steps)
        test_dataset = Parity('test', FLAGS.rank, FLAGS.num_steps)

    if not FLAGS.infinite:
        dataset = FiniteWrapper(
            dataset,
            FLAGS.dataset,
            FLAGS.capacity,
            FLAGS.rank,
            FLAGS.num_steps)

    shuffle = True
    sampler = None

    if world_size > 1:
        group = dist.init_process_group(
            backend='nccl',
            init_method='tcp://localhost:8113',
            world_size=world_size,
            rank=rank_idx,
            group_name="default")

    torch.cuda.set_device(rank)
    device = torch.device('cuda')

    FLAGS_OLD = FLAGS

    # Load model and key arguments
    if FLAGS.resume_iter != 0:
        # resume (tiep tuc train)  -> model_latest.pth  (trang thai moi nhat)
        # eval-only (--train tat)  -> model_best.pth neu co (performance tot nhat)
        ckpt_name = "model_latest.pth"
        if not FLAGS_OLD.train and osp.exists(osp.join(logdir, "model_best.pth")):
            ckpt_name = "model_best.pth"
        model_path = osp.join(logdir, ckpt_name)
        print(f"[INFO] load checkpoint: {model_path}")

        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(model_path, weights_only=True)

        # Khoi phuc moc best de model_best.pth khong bi ghi de boi ket qua te hon.
        best_path = osp.join(logdir, "model_best.pth")
        if osp.exists(best_path):
            try:
                with torch.serialization.safe_globals([argparse.Namespace]):
                    _b = torch.load(best_path, weights_only=True)
                if _b.get('best_oracle_error') is not None:
                    best_oracle_error = _b['best_oracle_error']
                    print(f"[INFO] best_oracle_error truoc do = {best_oracle_error:.3e}")
            except Exception:
                pass

        FLAGS = checkpoint['FLAGS']

        FLAGS.resume_iter = FLAGS_OLD.resume_iter
        FLAGS.save_interval = FLAGS_OLD.save_interval
        FLAGS.gpus = FLAGS_OLD.gpus
        FLAGS.train = FLAGS_OLD.train
        FLAGS.batch_size = FLAGS_OLD.batch_size
        FLAGS.step_lr = FLAGS_OLD.step_lr
        FLAGS.num_steps = FLAGS_OLD.num_steps
        FLAGS.exp = FLAGS_OLD.exp
        FLAGS.ponder = FLAGS_OLD.ponder
        FLAGS.hopfield = getattr(FLAGS_OLD, 'hopfield', False)
        FLAGS.beta = getattr(FLAGS_OLD, 'beta', None)
        FLAGS.truncate_hopfield = getattr(FLAGS_OLD, 'truncate_hopfield', False)
        FLAGS.heatmap = getattr(FLAGS_OLD, 'heatmap', False)
        FLAGS.num_heads = getattr(FLAGS_OLD, 'num_heads', 8)
        FLAGS.tie_mode = getattr(FLAGS_OLD, 'tie_mode', 'hard')
        FLAGS.tie_gamma = getattr(FLAGS_OLD, 'tie_gamma', 0.01)
        FLAGS.deep_sup = getattr(FLAGS_OLD, 'deep_sup', False)
        FLAGS.test_steps = getattr(FLAGS_OLD, 'test_steps', 20)
        FLAGS.num_iterations = getattr(FLAGS_OLD, 'num_iterations', 10000)
        FLAGS.log_interval = getattr(FLAGS_OLD, 'log_interval', 10)
        FLAGS.cuda = getattr(FLAGS_OLD, 'cuda', False)

        model, optimizer = init_model(FLAGS, device, dataset)
        state_dict = model.state_dict()

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        model, optimizer = init_model(FLAGS, device, dataset)

    if FLAGS.gpus > 1:
        sync_model(model)

    print("num_parameters: ", sum([p.numel() for p in model.parameters()]))

    train_dataloader = DataLoader(
        dataset,
        num_workers=FLAGS.data_workers,
        batch_size=FLAGS.batch_size,
        shuffle=shuffle,
        pin_memory=False,
        worker_init_fn=worker_init_fn)
    def _make_test_loader(ds):
        if ds is None:
            return None
        return DataLoader(
            ds,
            num_workers=FLAGS.data_workers,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            worker_init_fn=worker_init_fn)

    test_loader_id = _make_test_loader(test_dataset)
    test_loader_ood = _make_test_loader(test_dataset_ood)

    logger = SummaryWriter(logdir)
    it = FLAGS.resume_iter

    FLAGS.use_wandb = not getattr(FLAGS, 'no_wandb', False)

    if FLAGS.use_wandb and rank_idx == 0:
        if not HAS_WANDB:
            raise RuntimeError(
                "\n[ERROR] Thư viện 'wandb' chưa được cài đặt!\n"
                "Vui lòng cài đặt bằng lệnh: pip install wandb\n"
                "(Hoặc truyền cờ --no_wandb nếu bạn muốn chạy mà không cần W&B)."
            )
        run_name = FLAGS.run_name if FLAGS.run_name else FLAGS.exp
        wandb.init(
            project=getattr(FLAGS, 'wandb_project', 'irem-experiments'),
            entity=getattr(FLAGS, 'wandb_entity', None),
            name=run_name,
            config=vars(FLAGS)
        )

    if FLAGS.train:
        model.train()
    else:
        model.eval()

    if FLAGS.train:
        train(
            train_dataloader,
            test_loader_id,
            test_loader_ood,
            logger,
            model,
            optimizer,
            FLAGS,
            logdir,
            rank_idx)
    else:
        test(test_loader_id, test_loader_ood, model, FLAGS,
             step=FLAGS.resume_iter)

    if getattr(FLAGS, 'use_wandb', False) and rank_idx == 0 and HAS_WANDB:
        try:
            wandb.finish()
        except Exception:
            pass


def main():
    FLAGS = parser.parse_args()
    FLAGS.replay_buffer = not FLAGS.no_replay_buffer
    logdir = osp.join(FLAGS.logdir, FLAGS.exp)

    if FLAGS.recurrent:
        FLAGS.no_replay_buffer = True

    if FLAGS.decoder:
        FLAGS.no_replay_buffer = True

    if not osp.exists(logdir):
        os.makedirs(logdir)

    if FLAGS.gpus > 1:
        mp.spawn(main_single, nprocs=FLAGS.gpus, args=(FLAGS,))
    else:
        main_single(0, FLAGS)


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except BaseException:
        pass

    main()
