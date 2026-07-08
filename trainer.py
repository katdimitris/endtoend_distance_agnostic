import json
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from utils import get_random_ds
from utils import ErrorCounter, OneHotToInt
from utils import *

class Trainer():
    def __init__(self, models, parameters, writer, exp_dir, exp_name):
        self.exp_dir = exp_dir
        self.exp_name = exp_name
        self.writer = writer

        self.models = models
        self.transmitter = models['transmitter']
        self.channel = models['channel']
        self.receiver = models['receiver']

        self.parameters = parameters
        self.device = parameters['device']
        self.loader = get_random_ds(size=parameters['train_size'],
                                    max_num=parameters['M'],
                                    N_symbols=parameters['N'],
                                    device=self.device,
                                    batch_size=parameters['batch'],
                                    dtype=T.float)
        self.eval_loader = get_random_ds(size=parameters['val_size'],
                                    max_num=parameters['M'],
                                    N_symbols=parameters['N'],
                                    device=self.device,
                                    batch_size=parameters['batch'],
                                    dtype=T.float)

        self.train_iter = parameters['train_iter']
        self.effective_bs = parameters['batch'] // parameters['N']

        self.train_km_tensor = None
        if parameters.get('train_km_list'):
            self.train_km_tensor = T.tensor(parameters['train_km_list'], dtype=T.float, device=self.device)

        if 'film' in parameters["receiver_type"]:
            base_lr = parameters["lr"]
            film_lr = base_lr  # e.g. 10x smaller
            backbone_params = (
                    list(self.transmitter.parameters()) +
                    self.receiver.non_film_parameters()
            )
            film_params = self.receiver.film_parameters()
            self.optimizer = T.optim.Adam(
                [
                    {"params": backbone_params, "lr": base_lr},
                    {"params": film_params, "lr": film_lr},
                ]
            )
        else:
            params = list(self.transmitter.parameters()) + list(self.receiver.parameters())

            self.optimizer = T.optim.Adam(params, lr=parameters['lr'])
    
        self.scheduler = T.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=parameters['train_iter'])
        self.criterion = nn.CrossEntropyLoss()

        self.error_counter = ErrorCounter(parameters['bits'])
        self.onehot_to_int = OneHotToInt()

    def run(self):
        self.transmitter.train()
        self.channel.train()
        self.receiver.train()

        loss_meter = AverageMeter()
        errors, bits_counter = 0, 0
        loader_iter = iter(self.loader)


        for step in tqdm(range(0, self.train_iter)):
            lengths = self.sample_uniform_lengths()
            try:
                inputs, targets = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.loader)
                inputs, targets = next(loader_iter)
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.channel.set_length(lengths)

            t_outputs = self.transmitter(inputs, lengths.repeat_interleave(11))
            c_outputs = self.channel(t_outputs)
            r_outputs = self.receiver(c_outputs, lengths)

            loss = self.criterion(r_outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            loss_meter.update(loss.item(), inputs.size(0))
            errors_t, bits = self.error_counter(self.onehot_to_int(r_outputs.detach()), targets)
            errors += errors_t
            bits_counter += bits
            if step % self.parameters['log_interval'] == 0 and self.parameters['logging']:
                self.writer.add_scalar(f'Train/LR', self.optimizer.param_groups[0]['lr'], step)
                self.writer.add_scalar(f'Train/CE Loss', loss_meter.avg, step)
                self.writer.add_scalar(f'Train/BLER', errors.astype(np.float64) / float(bits_counter), step)
                loss_meter = AverageMeter()
                errors, bits_counter = 0, 0

                self._log_grad_norms(self.transmitter, 'Transmitter', step)
                self._log_grad_norms(self.receiver, 'Receiver', step)
                if hasattr(self.transmitter, 'last_film_stats'):
                    self._log_film_stats(self.transmitter, 'Transmitter', step)
                if hasattr(self.receiver, 'last_film_stats'):
                    self._log_film_stats(self.receiver, 'Receiver', step)

            # if step % (self.parameters['log_interval']*10) == 0 and self.parameters['logging']:
            #     with T.no_grad():
            #         self.evaluate(length=None, step=step)

        if self.parameters.get('no_eval', False):
            return

        eval_step = self.parameters.get('eval_step', 1)
        with T.no_grad():
            for eval_length in tqdm(range(self.parameters['transfer_lengths'][0], self.parameters['transfer_lengths'][1]+1, eval_step)):
                bler, loss = self.evaluate(length=eval_length, step=None)
                self.writer.add_scalar("Final Val/BLER Pretraining", bler, eval_length)
                self.writer.add_scalar("Final Val/Loss Pretraining", loss, eval_length)

    def evaluate(self, length, step):
        self.transmitter.eval()
        self.channel.eval()
        self.receiver.eval()
        loss_meter = AverageMeter()
        errors, bits_counter = 0, 0
        all_lengths = False
        if length is None:
            all_lengths = True
        else:
            self.channel.set_length(length=length)
        for (inputs, targets) in self.eval_loader:
            if all_lengths:
                length = self.sample_uniform_lengths()
                self.channel.set_length(length)

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if isinstance(length, torch.Tensor):
                trans_length = length.repeat_interleave(11)
            else:
                trans_length = length
            t_outputs = self.transmitter(inputs, trans_length)
            c_outputs = self.channel(t_outputs, length)
            r_outputs = self.receiver(c_outputs, length) # length for attn only

            loss = self.criterion(r_outputs, targets)
            loss_meter.update(loss.item(), inputs.size(0))
            errors_t, bits = self.error_counter(self.onehot_to_int(r_outputs.detach()), targets)
            errors += errors_t
            bits_counter += bits

        bler = errors.astype(np.float64) / float(bits_counter)
        if step is not None:
            if all_lengths:
                self.writer.add_scalar(f'Val/CE Loss', loss_meter.avg, step)
                self.writer.add_scalar(f'Val/BLER', bler, step)
            else:
                length_str = f' @{length}km'
                self.writer.add_scalar(f'Val{length_str}/CE Loss', loss_meter.avg, step)
                self.writer.add_scalar(f'Val{length_str}/BLER', bler, step)

        return bler, loss_meter.avg


    def save(self, path=None):
        """Saves transmitter/channel/receiver state dicts, plus the `parameters` dict that
        defines the architecture (as parameters.pt, exact, and parameters.json, for humans),
        under checkpoints/exp_dir/exp_name (or an explicit `path`). Without the parameters
        record, a checkpoint is only loadable by someone who remembers (or can guess) the
        exact CLI flags used to train it -- inference.py's load_models() reads parameters.pt
        as the source of truth instead of trusting re-supplied flags to match."""
        if path is None:
            path = os.path.join('checkpoints', self.exp_dir, self.exp_name)
        os.makedirs(path, exist_ok=True)
        T.save(self.transmitter.state_dict(), os.path.join(path, 'transmitter.pt'))
        T.save(self.channel.state_dict(), os.path.join(path, 'channel.pt'))
        T.save(self.receiver.state_dict(), os.path.join(path, 'receiver.pt'))
        T.save(self.parameters, os.path.join(path, 'parameters.pt'))
        with open(os.path.join(path, 'parameters.json'), 'w') as f:
            json.dump({k: str(v) for k, v in self.parameters.items()}, f, indent=2, sort_keys=True)
        return path

    def _log_grad_norms(self, module, prefix, step):
        for name, submodule in module.named_children():
            sq_sum = 0.0
            found = False
            for p in submodule.parameters():
                if p.grad is not None:
                    sq_sum += p.grad.detach().float().norm(2).item() ** 2
                    found = True
            if found:
                self.writer.add_scalar(f'GradNorm/{prefix}/{name}', sq_sum ** 0.5, step)

    def _log_film_stats(self, module, prefix, step):
        # Diagnoses whether gamma/beta actually respond to length, or collapse to a
        # constant regardless of distance: std_across_batch is computed over the batch
        # dim (samples in a batch carry different lengths), so a value near zero means
        # the FiLM output isn't varying with length.
        for layer_idx, stats in module.last_film_stats.items():
            for name, tensor in stats.items():
                self.writer.add_scalar(f'FiLM/{prefix} L{layer_idx}/{name} mean', tensor.mean().item(), step)
                self.writer.add_scalar(f'FiLM/{prefix} L{layer_idx}/{name} std_across_batch',
                                        tensor.std(dim=0).mean().item(), step)

    def sample_uniform_lengths(self, N=None):
        if N is None:
            N = self.effective_bs

        if self.train_km_tensor is not None:
            idx = T.randint(0, self.train_km_tensor.numel(), (N,), device=self.device)
            return self.train_km_tensor[idx]

        min = self.parameters['train_lengths'][0]
        max = self.parameters['train_lengths'][1]
        return (min + (max - min) * T.rand(N, device=self.device))
