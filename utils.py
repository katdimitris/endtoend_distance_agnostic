import errno
import math
import os
import random
import argparse
from typing import Optional

import numpy as np
import torch as T
from torch.utils.data import DataLoader, Dataset


def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(x) for x in obj]
    else:
        return obj




def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def fix_seed(seed: int):
   random.seed(seed)
   np.random.seed(seed)
   T.manual_seed(seed)
   if T.cuda.is_available():
       T.cuda.manual_seed(seed)
       T.cuda.manual_seed_all(seed)
   T.backends.cudnn.deterministic = True
   T.backends.cudnn.benchmark = False


class AverageMeter(object):
   def __init__(self):
       self.reset()

   def reset(self):
       self.val = 0
       self.avg = 0
       self.sum = 0
       self.count = 0

   def update(self, val, n=1):
       self.val = val
       self.sum += val * n
       self.count += n
       self.avg = self.sum / self.count


def get_closest_tag(var_length, parameters=None):
    if parameters['tags_encoding'] == 'one_hot':
        tags = np.arange(parameters['train_lengths'][0],
                         parameters['train_lengths'][1] + parameters['tags_n'],
                         parameters['tags_n'])
        closest_idx = np.argmin(np.abs(np.array(tags) - var_length))
        one_hot_tags = np.eye(len(tags))
        encoded_tags = one_hot_tags[closest_idx]
    elif parameters['tags_encoding'] == 'gaussian':
        range_start = parameters['train_lengths'][0]
        range_end = parameters['train_lengths'][1]
        dim = parameters['tags_d']
        sigma = parameters['tags_s']
        continuous_pos = (var_length - range_start) / (range_end - range_start) * (dim - 1)
        indices = np.arange(dim)
        gaussian = np.exp(-0.5 * ((indices - continuous_pos) / sigma) ** 2)
        # gaussian /= np.sum(gaussian)
        encoded_tags = gaussian
    else:
        raise ValueError("Unsupported encoding type")

    return T.tensor(encoded_tags, dtype=T.float32)


def gaussian_embed(x, start=30.0, end=50.0, d=10, sigma=2.0, normalize=True):
    """
    Returns a 1D tensor of shape [d], representing a Gaussian-like embedding
    for 'x' in [start, end].
      - Maps x into [0, d-1] so dimension i is roughly "distance" from x.
      - embed[i] = exp(-0.5 * ((i - scaled_x)/sigma)^2).
      - If normalize=True, divides by the sum so it sums to 1.
    """
    scaled_x = (x - start) / (end - start) * (d - 1)
    i = T.arange(d, dtype=T.float)
    emb = T.exp(-0.5 * ((i - scaled_x) / sigma) ** 2)
    if normalize:
        emb = emb / emb.sum()
    return emb

def build_reference_table(start=30.0, end=50.0, intervals=40, d=10, sigma=2.0, normalize=True):
    """
    Returns a tensor of shape [intervals, d],
    each row is the Gaussian embedding for one reference length.
    """
    points = T.linspace(start, end, intervals)
    embs = [gaussian_embed(x, start, end, d, sigma, normalize) for x in points]
    return T.stack(embs, dim=0)


class ClippingReLU(T.nn.Module):
    def __init__(self, s_q):
        super().__init__()
        self.epsilon = s_q / 2.0

    def forward(self, x: T.Tensor):
        return T.relu(x - self.epsilon) - T.relu(x - (math.pi / 4.0) +
                                                 self.epsilon)


class PhotonicSigmoid(T.nn.Module):
    def __init__(self, A1=0.060, A2=1.005, x0=0.145, d=0.033, cutoff=2):
        super().__init__()
        self.A1 = A1
        self.A2 = A2
        self.x0 = x0
        self.d = d
        self.cutoff = cutoff

    def forward(self, x: T.Tensor):
        x = x - self.x0
        x.clamp_(max=self.cutoff)
        return self.A2 + (self.A1 - self.A2) / (1 + T.exp(x / self.d))


class PhotonicSinusoidal(T.nn.Module):
    def __init__(self, x_lower=0, x_upper=1):
        super().__init__()
        self.x_lower = x_lower
        self.x_upper = x_upper
        self.y_upper = 2

    def forward(self, x: T.Tensor):
        x = x.clamp(self.x_lower, self.x_upper)
        return T.pow(T.sin(x * math.pi / 2.0), self.y_upper)


def signal_normalize_min_max(sig, min, max, axis=1):
    ar = (sig - T.min(sig)) * (max - min)
    par = T.max(sig) - T.min(sig)
    norm_sig = min + (ar / par)
    return norm_sig


def signal_normalize(sig, axis=1):
    mean = -T.mean(sig, axis=axis, keepdim=True)
    nodc_sig = T.add(sig, mean)
    z = T.abs(nodc_sig)
    z = T.pow(z, 2)
    z = T.mean(z, axis, keepdim=True)
    z = T.sqrt(z)
    norm_sig = T.div(nodc_sig, z)
    return norm_sig


class IntToBits(object):
    def __init__(self, max_num=64):
        self.max_num = 64
        pass

    def __call__(self, x):
        return np.unpackbits(x.astype(np.uint8))


class OneHotToInt(T.nn.Module):
    def __init__(self):
        super(OneHotToInt, self).__init__()

    def forward(self, x):
        return T.argmax(x, axis=1)


class ErrorCounter(object):
    def __init__(self, bits_size=6):
        self.bits_size = bits_size

    def __call__(self, output, target):
        t_target = target.detach().cpu().numpy().astype(np.uint8)
        t_output = output.detach().cpu().numpy().astype(np.uint8)

        target_bits = np.unpackbits(t_target).reshape(-1,
                                                      8)[:,
                                                         (8 - self.bits_size):]

        output_bits = np.unpackbits(t_output).reshape(-1,
                                                      8)[:,
                                                         (8 - self.bits_size):]

        output_bits = output_bits.reshape(-1)
        target_bits = target_bits.reshape(-1)

        absulute = np.abs(
            output_bits.astype(np.int32) - target_bits.astype(np.int32))

        errors = np.sum(absulute)

        return errors, output_bits.shape[0]


class BLER(object):
    def __init__(self, size=6):
        self.error_counter = ErrorCounter(size)
        pass

    def __call__(self, output, target):
        errors, len_output_bits = self.error_counter(output, target)
        return errors * 1.0 / (len_output_bits * 1.0)


def roll(x: T.Tensor,
         shift: int,
         dim: int = -1,
         fill_pad: Optional[int] = None):

    device = x.device

    if 0 == shift:
        return x

    elif shift < 0:
        shift = -shift
        gap = x.index_select(dim, T.arange(shift, device=device))
        if fill_pad is not None:
            gap = fill_pad * T.ones_like(gap, device=device)
        return T.cat([
            x.index_select(dim, T.arange(shift, x.size(dim), device=device)),
            gap
        ],
                     dim=dim)

    else:
        shift = x.size(dim) - shift
        gap = x.index_select(dim, T.arange(shift, x.size(dim), device=device))
        if fill_pad is not None:
            gap = fill_pad * T.ones_like(gap, device=device)
        return T.cat(
            [gap, x.index_select(dim, T.arange(shift, device=device))],
            dim=dim)


def ifnot_create(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return path


class OneHotVecGenerator(object):
    def __init__(self):
        pass

    def __call__(self, max_num=64, samples_num=1000):
        self.max_num = max_num
        sample = np.random.randint(0, max_num, size=samples_num)
        return T.from_numpy(sample)

    def convert_to_onehot(self, x):
        if (len(x.size()) != 0):
            onehot = T.zeros((x.size()[0], self.max_num))
            onehot[T.arange(x.size()[0]), x.type(T.long)] = 1
        else:
            onehot = T.zeros(self.max_num)
            onehot[x] = 1
        return onehot


def custom_collate_fn(data, N_symbols):
    batch_size = len(data)
    inputs, targets = zip(*data)

    targets = T.tensor([
        targets[i]
        for i in [i for i in range(N_symbols // 2, batch_size, N_symbols)]
    ])

    inputs = T.stack(list(inputs), dim=0)

    return inputs, targets


class RandomDataset(Dataset):
    def __init__(self, max_num, N_symbols, device, samples_num, dtype=T.double):
        self.generator = OneHotVecGenerator()
        self.max_num = max_num
        self.samples_num = samples_num
        self.dtype = dtype
        self.N_symbols = N_symbols
        self.data = self.generator(max_num, samples_num).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.generator.convert_to_onehot(
            self.data[i]), self.data[(i // self.N_symbols) * self.N_symbols +
                                     (self.N_symbols // 2)].type(T.long)

    def get_onehot(self):
        return self.generator.convert_to_onehot(self.data)

    def get_data(self):
        return self.data


def get_random_ds(size, max_num, N_symbols, device, batch_size=64, dtype=T.double):
    return DataLoader(
        RandomDataset(max_num, N_symbols, device, size, dtype),
        batch_size,
        collate_fn=lambda data: custom_collate_fn(data, N_symbols),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True)
