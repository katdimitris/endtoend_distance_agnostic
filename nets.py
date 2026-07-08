import os

import torch
import torch as T
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from utils import PhotonicSigmoid


class DistanceQueryReceiver2(nn.Module):
    def __init__(self,
                 n_input=48,      # dim of channel output (c_outputs)
                 n_output=64,     # number of messages/classes M
                 d_model=64,
                 n_tokens=4,
                 n_heads=1,
                 L_min=10.0,
                 L_max=40.0):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.L_min = L_min
        self.L_max = L_max

        # 1) Project single channel vector y into a sequence of n_tokens latent tokens
        # y: [B, n_input] -> H: [B, n_tokens, d_model]
        self.y_to_tokens = nn.Linear(n_input, n_tokens * d_model)

        # 2) Embed normalized length scalar into query vector
        # length_norm in [0,1] -> [B, d_model]
        self.length_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 3) Cross-attention: query = distance, key/value = signal tokens
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True  # we use [B, L, D]
        )

        # Normalization + residual around attention
        self.norm_tokens = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model)

        # 4) Feed-forward block (as in Transformers) with residual
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm_ff = nn.LayerNorm(d_model)

        # 5) Final classifier on distance-conditioned representation
        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_output)
        )

    def forward(self, y, length):
        """
        y:      [B, n_input]   channel output from self.channel(...)
        length: scalar or tensor, in km (1040)
        returns: logits [B, n_output]
        """
        B = y.size(0)
        device = y.device
        dtype = y.dtype

        # --- 1) y -> latent tokens ------------------------------------------
        # Shape: [B, n_tokens * d_model] -> [B, n_tokens, d_model]
        H = self.y_to_tokens(y).view(B, self.n_tokens, self.d_model)
        # Normalize tokens for stability
        H = self.norm_tokens(H)

        # --- 2) length -> normalized scalar -> query embedding -------------
        if not torch.is_tensor(length):
            length = torch.tensor(length, device=device, dtype=dtype)
        length = length.to(device=device, dtype=dtype)

        # make length shape [B]
        if length.dim() == 0:
            length = length.unsqueeze(0).expand(B)
        elif length.dim() == 1 and length.size(0) == 1:
            length = length.expand(B)
        # else assume [B]

        # normalize L into [0,1]
        L_norm = (length - self.L_min) / (self.L_max - self.L_min)
        L_norm = L_norm.clamp(0.0, 1.0).view(B, 1)  # [B, 1]

        # embed to query space
        q = self.length_embed(L_norm)               # [B, d_model]
        q = q.unsqueeze(1)                          # [B, 1, d_model]

        # --- 3) cross-attention: Q = distance, K/V = tokens from y ---------
        # attention expects queries, keys, values of shape [B, L, D]
        # Q: [B, 1, D], K,V: [B, T, D]
        attn_out, attn_weights = self.attn(q, H, H)  # [B, 1, D]

        # residual connection: add original q
        x = q + attn_out                             # [B, 1, D]
        x = self.norm_attn(x)

        # --- 4) position-wise feed-forward + residual ----------------------
        ff_out = self.ff(x)                          # [B, 1, D]
        x = self.norm_ff(x + ff_out)                 # [B, 1, D]

        # --- 5) classify using the single distance-conditioned vector ------
        x = x.squeeze(1)                             # [B, D]
        logits = self.cls(x)                         # [B, n_output]

        return logits

class DistanceQueryReceiver(nn.Module):
    def __init__(self,
                 n_input=48,
                 n_output=64,
                 d_model=128,
                 n_tokens=1,
                 n_heads=1,
                 L_min=10.0,
                 L_max=40.0):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.L_min = L_min
        self.L_max = L_max

        self.y_to_tokens = nn.Linear(n_input, n_tokens * d_model)

        # length (normalized scalar) -> d_model
        self.length_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_output)
        )

    def forward(self, y, length):
        B = y.size(0)
        device, dtype = y.device, y.dtype

        # --- 1) y -> tokens ---
        H = self.y_to_tokens(y).view(B, self.n_tokens, self.d_model)

        # --- 2) length -> normalized scalar -> embedding ---
        if not torch.is_tensor(length):
            length = torch.tensor(length, device=device, dtype=dtype)
        length = length.to(device=device, dtype=dtype)

        if length.dim() == 0:
            length = length.unsqueeze(0).expand(B)
        elif length.dim() == 1 and length.size(0) == 1:
            length = length.expand(B)

        # normalize to [0, 1]
        L_norm = (length - self.L_min) / (self.L_max - self.L_min)
        L_norm = L_norm.clamp(0.0, 1.0).view(B, 1)  # [B, 1]

        q = self.length_embed(L_norm).unsqueeze(1)  # [B, 1, d_model]

        # --- 3) cross-attention ---
        out, attn_weights = self.attn(q, H, H)
        out = out.squeeze(1)                        # [B, d_model]

        # --- 4) classify ---
        logits = self.cls(out)
        return logits

class Receiver(nn.Module):
    def __init__(self, n_input=48, hidden=128, n_output=64):
        super().__init__()

        self.layer_in = nn.Linear(n_input, hidden)
        self.layer_1 = nn.Linear(hidden, hidden)
        self.layer_out = nn.Linear(hidden, n_output)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, x, length=None):
        x = self.layer_in(x)
        x = self.act1(x)
        x = self.layer_1(x)
        x = self.act2(x)
        x = self.layer_out(x)

        return x

    def forward_features(self, x: T.Tensor) -> T.Tensor:
        x = self.layer_in(x)
        x = self.act(x)
        x = self.layer_1(x)
        x = self.act_1(x)

        return x


class Transimitter(nn.Module):
    def __init__(self, n_input=64, hidden=128, n_output=48):
        super().__init__()

        self.layer_in = nn.Linear(n_input, hidden)
        self.layer_1 = nn.Linear(hidden, hidden)
        self.layer_out = nn.Linear(hidden, n_output)
        self.act = torch.nn.ReLU()
        # self.act_out = PhotonicSigmoid
        self.act_out = torch.nn.Sigmoid()

    def forward(self, x, lengths):
        x = self.layer_in(x)
        x = self.act(x)
        x = self.layer_1(x)
        x = self.act(x)
        x = self.layer_out(x)
        x = self.act_out(x)

        return x

import math
import torch
import torch.nn as nn




class FiLMTransimitter(nn.Module):
    def __init__(
        self,
        n_input=64,
        hidden=128,
        n_output=48,
        min_length=1.0,
        max_length=60.0,
        film_layers=(1,),
        length_encoding="scalar",
        position="preact",
        length_step=2.0,
        film_mode="both",
        film_trunk_layers=2,
        film_trunk_hidden=None,
    ):
        """
        Same FiLM interface as FiLMReceiver, but keeps the original Transimitter
        backbone dimensions: 64 -> 128 -> 128 -> 48 with ReLU/ReLU/Sigmoid.
        """
        super().__init__()


        # Keep original transmitter dims
        self.n_input = n_input
        self.hidden = hidden
        self.n_output = n_output


        # Length / FiLM config (same as receiver)
        self.minL = float(min_length)
        self.maxL = float(max_length)


        self.length_encoding = str(length_encoding).lower()
        if self.length_encoding not in ("scalar", "one_hot"):
            raise ValueError(f"length_encoding must be 'scalar' or 'one_hot', got {length_encoding!r}")


        self.length_step = float(length_step)
        if self.length_encoding == "one_hot":
            if self.length_step <= 0:
                raise ValueError("length_step must be > 0 when length_encoding='one_hot'")
            self.num_bins = int(math.floor((self.maxL - self.minL) / self.length_step)) + 1
        else:
            self.num_bins = None


        self.film_mode = str(film_mode).lower()
        if self.film_mode not in ("both", "beta", "gamma"):
            raise ValueError(f"film_mode must be 'both', 'beta', or 'gamma', got {film_mode!r}")


        self.set_film_layers(film_layers)
        self.position = position  # "preact" matches receiver behavior
        self.last_film_stats = {}  # layer_idx -> {'gamma'/'beta': last batch's tensor}, for diagnostics


        # Backbone (unchanged)
        self.layer_in = nn.Linear(n_input, hidden)
        self.layer_1 = nn.Linear(hidden, hidden)
        self.layer_out = nn.Linear(hidden, n_output)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act_out = nn.Sigmoid()


        # ----- FiLM trunk (same pattern as receiver) -----
        trunk_in_dim = 1 if self.length_encoding == "scalar" else self.num_bins


        film_trunk_layers = int(film_trunk_layers)
        if film_trunk_layers < 1:
            raise ValueError("film_trunk_layers must be >= 1")


        trunk_width = hidden if film_trunk_hidden is None else int(film_trunk_hidden)
        if trunk_width < 1:
            raise ValueError("film_trunk_hidden must be >= 1")


        self.film_trunk_layers = film_trunk_layers
        self.film_trunk_hidden = trunk_width


        self.length_trunk = self._make_mlp(
            in_dim=trunk_in_dim,
            hidden_dim=trunk_width,
            out_dim=hidden,               # embed dim feeding the heads
            num_linear_layers=film_trunk_layers,
            activation=nn.ReLU,
        )


        # FiLM heads (match layer dims: hidden, hidden, n_output)
        def head_out_dim(d):
            return 2 * d if self.film_mode == "both" else d


        self.film_head_0 = nn.Linear(hidden, head_out_dim(hidden))     # for layer_in output
        self.film_head_1 = nn.Linear(hidden, head_out_dim(hidden))     # for layer_1 output
        self.film_head_2 = nn.Linear(hidden, head_out_dim(n_output))   # for layer_out output


    @staticmethod
    def _make_mlp(in_dim, hidden_dim, out_dim, num_linear_layers, activation=nn.ReLU):
        layers = []
        L = int(num_linear_layers)


        if L == 1:
            layers.append(nn.Linear(in_dim, out_dim))
            return nn.Sequential(*layers)


        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation())


        for _ in range(L - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())


        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)


    def film_parameters(self):
        return (
            list(self.length_trunk.parameters()) +
            list(self.film_head_0.parameters()) +
            list(self.film_head_1.parameters()) +
            list(self.film_head_2.parameters())
        )


    def non_film_parameters(self):
        return (
            list(self.layer_in.parameters()) +
            list(self.layer_1.parameters()) +
            list(self.layer_out.parameters())
        )


    def set_film_layers(self, film_layers):
        film_layers = tuple(sorted(set(int(i) for i in film_layers)))
        for i in film_layers:
            if i not in (0, 1, 2):
                raise ValueError(f"film_layers can only contain 0,1,2. Got {i}.")
        self.film_layers = film_layers


    def _prep_length_scalar(self, length, B, device, dtype):
        if not torch.is_tensor(length):
            length = torch.tensor(length, device=device, dtype=dtype)
        else:
            length = length.to(device=device, dtype=dtype)


        if length.dim() == 0:
            length = length.unsqueeze(0).expand(B)
        elif length.dim() == 1 and length.numel() == 1:
            length = length.expand(B)


        L_norm = (length - self.minL) / (self.maxL - self.minL)
        return L_norm.clamp(0.0, 1.0).view(B, 1)


    def _prep_length_one_hot(self, length, B, device, dtype):
        if not torch.is_tensor(length):
            length = torch.tensor(length, device=device, dtype=dtype)
        else:
            length = length.to(device=device, dtype=dtype)


        if length.dim() == 0:
            length = length.unsqueeze(0).expand(B)
        elif length.dim() == 1 and length.numel() == 1:
            length = length.expand(B)


        length = length.clamp(self.minL, self.maxL)
        idx = torch.round((length - self.minL) / self.length_step).long()
        idx = idx.clamp(0, self.num_bins - 1)


        one_hot = torch.zeros(B, self.num_bins, device=device, dtype=dtype)
        one_hot.scatter_(1, idx.view(B, 1), 1.0)
        return one_hot


    def _apply_film(self, tensor, head_out, layer_idx):
        if self.film_mode == "both":
            gamma, beta = torch.chunk(head_out, 2, dim=-1)
            self.last_film_stats[layer_idx] = {"gamma": gamma.detach(), "beta": beta.detach()}
            return gamma * tensor + beta
        elif self.film_mode == "beta":
            self.last_film_stats[layer_idx] = {"beta": head_out.detach()}
            return tensor + head_out
        else:  # "gamma"
            self.last_film_stats[layer_idx] = {"gamma": head_out.detach()}
            return head_out * tensor


    def forward(self, x, length):
        """
        x: (B, n_input)
        length: scalar or (B,) or (B,1) depending on encoding
        """
        B = x.size(0)
        device, dtype = x.device, x.dtype


        if self.length_encoding == "scalar":
            L_in = self._prep_length_scalar(length, B, device, dtype)
        else:
            L_in = self._prep_length_one_hot(length, B, device, dtype)


        z = self.length_trunk(L_in)


        # Layer 0 (layer_in -> ReLU)
        if self.position == "preact":
            h1 = self.layer_in(x)
        else:
            h1 = self.act1(self.layer_in(x))


        if 0 in self.film_layers:
            h1 = self._apply_film(h1, self.film_head_0(z), 0)


        if self.position == "preact":
            h1 = self.act1(h1)


        # Layer 1 (layer_1 -> ReLU)
        if self.position == "preact":
            h2 = self.layer_1(h1)
        else:
            h2 = self.act2(self.layer_1(h1))


        if 1 in self.film_layers:
            h2 = self._apply_film(h2, self.film_head_1(z), 1)


        if self.position == "preact":
            h2 = self.act2(h2)


        # Output (layer_out -> Sigmoid)
        out = self.layer_out(h2)
        if 2 in self.film_layers:
            out = self._apply_film(out, self.film_head_2(z), 2)


        out = self.act_out(out)
        return out

import math
import torch
import torch.nn as nn

class FiLMReceiver(nn.Module):
    def __init__(
        self,
        n_input=48,
        hidden=128,
        n_output=64,
        min_length=1.0,
        max_length=60.0,
        film_layers=(1,),
        length_encoding="scalar",
        position="preact",
        length_step=2.0,
        film_mode="both",
        film_trunk_layers=2,        # <-- NEW: number of Linear layers in the FiLM trunk
        film_trunk_hidden=None,     # <-- NEW (optional): width inside trunk; default=hidden
    ):
        super().__init__()

        self.n_input = n_input
        self.hidden = hidden
        self.n_output = n_output
        self.minL = float(min_length)
        self.maxL = float(max_length)

        self.length_encoding = str(length_encoding).lower()
        if self.length_encoding not in ("scalar", "one_hot"):
            raise ValueError(f"length_encoding must be 'scalar' or 'one_hot', got {length_encoding!r}")

        self.length_step = float(length_step)
        if self.length_encoding == "one_hot":
            if self.length_step <= 0:
                raise ValueError("length_step must be > 0 when length_encoding='one_hot'")
            self.num_bins = int(math.floor((self.maxL - self.minL) / self.length_step)) + 1
        else:
            self.num_bins = None

        self.film_mode = str(film_mode).lower()
        if self.film_mode not in ("both", "beta", "gamma"):
            raise ValueError(f"film_mode must be 'both', 'beta', or 'gamma', got {film_mode!r}")

        self.set_film_layers(film_layers)
        self.position = position
        self.last_film_stats = {}  # layer_idx -> {'gamma'/'beta': last batch's tensor}, for diagnostics

        # Backbone
        self.layer_in = nn.Linear(n_input, hidden)
        self.layer_1 = nn.Linear(hidden, hidden)
        self.layer_out = nn.Linear(hidden, n_output)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        # ----- Build configurable FiLM trunk -----
        trunk_in_dim = 1 if self.length_encoding == "scalar" else self.num_bins

        film_trunk_layers = int(film_trunk_layers)
        if film_trunk_layers < 1:
            raise ValueError("film_trunk_layers must be >= 1")

        trunk_width = hidden if film_trunk_hidden is None else int(film_trunk_hidden)
        if trunk_width < 1:
            raise ValueError("film_trunk_hidden must be >= 1")

        self.film_trunk_layers = film_trunk_layers
        self.film_trunk_hidden = trunk_width

        self.length_trunk = self._make_mlp(
            in_dim=trunk_in_dim,
            hidden_dim=trunk_width,
            out_dim=hidden,                 # embed dim feeding the heads
            num_linear_layers=film_trunk_layers,
            activation=nn.ReLU,
        )

        # --- FiLM heads depend on mode ---
        def head_out_dim(d):
            return 2 * d if self.film_mode == "both" else d

        self.film_head_0 = nn.Linear(hidden, head_out_dim(hidden))
        self.film_head_1 = nn.Linear(hidden, head_out_dim(hidden))
        self.film_head_2 = nn.Linear(hidden, head_out_dim(n_output))

    @staticmethod
    def _make_mlp(in_dim, hidden_dim, out_dim, num_linear_layers, activation=nn.ReLU):
        """
        Builds:
          L=1: Linear(in -> out)
          L=2: Linear(in -> hidden) -> act -> Linear(hidden -> out)
          L=3: Linear(in -> hidden) -> act -> Linear(hidden -> hidden) -> act -> Linear(hidden -> out)
          ...
        """
        layers = []
        L = int(num_linear_layers)

        if L == 1:
            layers.append(nn.Linear(in_dim, out_dim))
            return nn.Sequential(*layers)

        # first
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation())

        # middle (L-2 blocks)
        for _ in range(L - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())

        # last
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def film_parameters(self):
        return (list(self.length_trunk.parameters()) +
                list(self.film_head_0.parameters()) +
                list(self.film_head_1.parameters()) +
                list(self.film_head_2.parameters()))

    def non_film_parameters(self):
        return (list(self.layer_in.parameters()) +
                list(self.layer_1.parameters()) +
                list(self.layer_out.parameters()))

    def set_film_layers(self, film_layers):
        film_layers = tuple(sorted(set(int(i) for i in film_layers)))
        for i in film_layers:
            if i not in (0, 1, 2):
                raise ValueError(f"film_layers can only contain 0,1,2. Got {i}.")
        self.film_layers = film_layers

    def _prep_length_scalar(self, length, B, device, dtype):
        if not torch.is_tensor(length):
            length = torch.tensor(length, device=device, dtype=dtype)
        else:
            length = length.to(device=device, dtype=dtype)

        if length.dim() == 0:
            length = length.unsqueeze(0).expand(B)
        elif length.dim() == 1 and length.numel() == 1:
            length = length.expand(B)

        L_norm = (length - self.minL) / (self.maxL - self.minL)
        return L_norm.clamp(0.0, 1.0).view(B, 1)

    def _prep_length_one_hot(self, length, B, device, dtype):
        if not torch.is_tensor(length):
            length = torch.tensor(length, device=device, dtype=dtype)
        else:
            length = length.to(device=device, dtype=dtype)

        if length.dim() == 0:
            length = length.unsqueeze(0).expand(B)
        elif length.dim() == 1 and length.numel() == 1:
            length = length.expand(B)

        length = length.clamp(self.minL, self.maxL)
        idx = torch.round((length - self.minL) / self.length_step).long()
        idx = idx.clamp(0, self.num_bins - 1)

        one_hot = torch.zeros(B, self.num_bins, device=device, dtype=dtype)
        one_hot.scatter_(1, idx.view(B, 1), 1.0)
        return one_hot

    def _apply_film(self, tensor, head_out, layer_idx):
        if self.film_mode == "both":
            gamma, beta = torch.chunk(head_out, 2, dim=-1)
            self.last_film_stats[layer_idx] = {"gamma": gamma.detach(), "beta": beta.detach()}
            return gamma * tensor + beta
        elif self.film_mode == "beta":
            self.last_film_stats[layer_idx] = {"beta": head_out.detach()}
            return tensor + head_out
        else:  # "gamma"
            self.last_film_stats[layer_idx] = {"gamma": head_out.detach()}
            return head_out * tensor

    def forward(self, x, length):
        B = x.size(0)
        device, dtype = x.device, x.dtype

        if self.length_encoding == "scalar":
            L_in = self._prep_length_scalar(length, B, device, dtype)
        else:
            L_in = self._prep_length_one_hot(length, B, device, dtype)

        z = self.length_trunk(L_in)

        # Layer 0
        if self.position == "preact":
            h1 = self.layer_in(x)
        else:
            h1 = self.act1(self.layer_in(x))

        if 0 in self.film_layers:
            h1 = self._apply_film(h1, self.film_head_0(z), 0)

        if self.position == "preact":
            h1 = self.act1(h1)

        # Layer 1
        if self.position == "preact":
            h2 = self.layer_1(h1)
        else:
            h2 = self.act2(self.layer_1(h1))

        if 1 in self.film_layers:
            h2 = self._apply_film(h2, self.film_head_1(z), 1)

        if self.position == "preact":
            h2 = self.act2(h2)

        out = self.layer_out(h2)
        if 2 in self.film_layers:
            out = self._apply_film(out, self.film_head_2(z), 2)

        return out






class ReceiverConcatLengthTags(nn.Module):
    def __init__(self,
                 n_input=48,
                 hidden=128,
                 n_output=64,
                 min_length=1.0,
                 max_length=60.0,
                 step=2.0,
                 length_encoding="one_hot",   # "one_hot" | "gaussian"
                 d=20,                       # gaussian embedding dimension
                 sigma=2.0,                  # gaussian std (in embedding-index units)
                 gaussian_normalize=True):   # normalize tags to sum=1
        super().__init__()

        self.n_input = n_input
        self.hidden = hidden
        self.n_output = n_output
        self.minL = float(min_length)
        self.maxL = float(max_length)
        self.step = float(step)

        self.length_encoding = str(length_encoding).lower()
        assert self.length_encoding in ("one_hot", "gaussian"), \
            f"length_encoding must be 'one_hot' or 'gaussian', got {length_encoding}"

        self.d = int(d)
        self.sigma = float(sigma)
        self.gaussian_normalize = bool(gaussian_normalize)

        if self.length_encoding == "one_hot":
            # Discrete length values (bins) for one-hot encoding
            length_values = torch.arange(min_length,
                                         max_length + 0.5 * step,  # include max
                                         step)
            self.register_buffer("length_values", length_values)  # [K]
            self.n_tags = length_values.numel()
        else:
            # Gaussian encoding uses an embedding axis of size d
            emb_idx = torch.arange(self.d, dtype=torch.float32)  # [d]
            self.register_buffer("gauss_idx", emb_idx)
            self.n_tags = self.d

        # First layer takes x plus length tags (one-hot or gaussian)
        self.layer_in  = nn.Linear(n_input + self.n_tags, hidden)
        self.layer_1   = nn.Linear(hidden, hidden)
        self.layer_out = nn.Linear(hidden, n_output)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def _make_length_batch(self, x, length):
        """Return length as a [B] tensor on x.device/x.dtype."""
        B = x.size(0)
        device = x.device
        dtype = x.dtype

        if not torch.is_tensor(length):
            length = torch.tensor(length, device=device, dtype=dtype)
        else:
            length = length.to(device=device, dtype=dtype)

        if length.dim() == 0:
            length = length.unsqueeze(0).expand(B)
        elif length.dim() == 1 and length.size(0) == 1:
            length = length.expand(B)
        # else assume [B]
        return length

    def _encode_one_hot(self, length, device, dtype):
        B = length.size(0)
        lv = self.length_values.to(device=device, dtype=dtype)      # [K]
        diff = (length.view(B, 1) - lv.view(1, -1)).abs()           # [B, K]
        idx = diff.argmin(dim=1)                                    # [B]
        tags = torch.zeros(B, self.n_tags, device=device, dtype=dtype)
        tags.scatter_(1, idx.unsqueeze(1), 1.0)
        return tags

    def _encode_gaussian(self, length, device, dtype):
        """
        Soft Gaussian encoding centered at the *relative position* of `length`
        within [minL, maxL], mapped onto indices [0, d-1].

        pos = (length - minL)/(maxL-minL) * (d-1)
        tags[j] = exp(-(j-pos)^2 / (2*sigma^2))
        """
        B = length.size(0)

        # Clamp to training range to avoid extreme means
        length_clamped = length.clamp(min=self.minL, max=self.maxL)

        denom = max(self.maxL - self.minL, 1e-12)
        pos = (length_clamped - self.minL) / denom * (self.d - 1)   # [B]

        j = self.gauss_idx.to(device=device, dtype=dtype).view(1, -1)  # [1, d]
        mu = pos.view(B, 1)                                             # [B, 1]

        sigma = max(self.sigma, 1e-12)
        tags = torch.exp(-0.5 * ((j - mu) / sigma) ** 2)               # [B, d]

        if self.gaussian_normalize:
            tags = tags / (tags.sum(dim=1, keepdim=True) + 1e-12)

        return tags

    def forward(self, x, length):
        """
        x      : [B, n_input] channel output
        length : scalar or [B] length in km
        """
        B = x.size(0)
        device = x.device
        dtype = x.dtype

        length = self._make_length_batch(x, length)

        if self.length_encoding == "one_hot":
            tags = self._encode_one_hot(length, device, dtype)       # [B, K]
        else:
            tags = self._encode_gaussian(length, device, dtype)      # [B, d]

        x_cat = torch.cat([x, tags], dim=-1)                         # [B, n_input + n_tags]

        h1 = self.act1(self.layer_in(x_cat))
        h2 = self.act2(self.layer_1(h1))
        out = self.layer_out(h2)                                     # [B, n_output]
        return out


class ReceiverTagsPlusFiLM(nn.Module):
    def __init__(self,
                 n_input=48,
                 hidden=128,
                 n_output=64,
                 min_length=1.0,
                 max_length=60.0,
                 step=2.0):
        super().__init__()

        self.n_input = n_input
        self.hidden = hidden
        self.n_output = n_output
        self.minL = min_length
        self.maxL = max_length
        self.step = step

        # --- discrete bins for one-hot tags ---
        length_values = torch.arange(min_length,
                                     max_length + 0.5 * step,
                                     step)
        self.register_buffer("length_values", length_values)  # [L_bins]
        self.n_tags = length_values.numel()

        # === main 3-layer MLP, but input is [x, tags] ===
        self.layer_in  = nn.Linear(n_input + self.n_tags, hidden)
        self.layer_1   = nn.Linear(hidden, hidden)
        self.layer_out = nn.Linear(hidden, n_output)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        # === FiLM: continuous length → gamma, beta on last hidden ===
        self.length_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * hidden)
        )

    def _one_hot_tags(self, length, B, device, dtype):
        # length: [B], self.length_values: [L_bins]
        lv = self.length_values.to(device=device, dtype=dtype)      # [L_bins]
        diff = (length.view(B, 1) - lv.view(1, -1)).abs()           # [B,L_bins]
        idx = diff.argmin(dim=1)                                    # [B]

        tags = torch.zeros(B, self.n_tags, device=device, dtype=dtype)
        tags.scatter_(1, idx.unsqueeze(1), 1.0)                     # one-hot
        return tags  # [B, n_tags]


    def forward(self, x, length):
        """
        x      : [B, n_input]   channel output
        length : scalar or [B] length in km
        """
        B = x.size(0)
        device = x.device
        dtype = x.dtype

        # --- normalize / broadcast length to [B] ---
        if not torch.is_tensor(length):
            length = torch.tensor(length, device=device, dtype=dtype)
        else:
            length = length.to(device=device, dtype=dtype)

        if length.dim() == 0:
            length = length.unsqueeze(0).expand(B)
        elif length.dim() == 1 and length.size(0) == 1:
            length = length.expand(B)
        # else: assume [B]

        # --- discrete one-hot tags for input ---
        tags = self._one_hot_tags(length, B, device, dtype)  # [B, n_tags]
        x_cat = torch.cat([x, tags], dim=-1)                 # [B, n_input + n_tags]

        # --- MLP backbone up to last hidden ---
        h1 = self.layer_in(x_cat)
        h1 = self.act1(h1)

        h2 = self.layer_1(h1)
        h2 = self.act2(h2)                                   # [B, hidden]

        # --- FiLM modulation on h2 using continuous length ---
        L_norm = (length - self.minL) / (self.maxL - self.minL)
        L_norm = L_norm.clamp(0.0, 1.0).view(B, 1)           # [B,1]

        gb = self.length_mlp(L_norm)                         # [B, 2*hidden]
        gamma, beta = torch.chunk(gb, 2, dim=-1)             # [B, hidden] each
        h2_mod = gamma * h2 + beta                           # [B, hidden]
        # --- final layer ---
        out = self.layer_out(h2_mod)                         # [B, n_output]
        return out





class SingleQueryCrossAttention(nn.Module):
    """
    Single-query cross-attention:
      - query_emb: [d] or [1, d]
      - context_embs: [N, d]
    """
    def __init__(self, d_model):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        self._init_as_diagonal(self.W_Q)
        self._init_as_diagonal(self.W_K)
        self._init_as_diagonal(self.W_V)

        self.softmax = T.nn.functional.softmax

    def _init_as_diagonal(self, linear_layer):
        with torch.no_grad():
            assert linear_layer.in_features == linear_layer.out_features, \
                "Only square linear layers can be initialized as diagonal"
            eye = torch.eye(linear_layer.in_features, device=linear_layer.weight.device)
            linear_layer.weight.data.copy_(eye)

    def forward(self, query_emb, context_embs):
        """
        Supports:
          - query_emb: [d] or [1, d], context_embs: [N, d]
          - query_emb: [B, d],       context_embs: [B, N, d]
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)  # [1, d]
            context_embs = context_embs.unsqueeze(0)  # [1, N, d]
        elif query_emb.dim() == 2 and context_embs.dim() == 2:
            # Single-query mode but both inputs are 2D
            context_embs = context_embs.unsqueeze(0)  # [1, N, d]
        elif query_emb.dim() != 2 or context_embs.dim() != 3:
            raise ValueError("query_emb must be [d] or [B, d], and context_embs must be [N, d] or [B, N, d]")

        Q = self.W_Q(query_emb).unsqueeze(1)  # [B, 1, d]
        K = self.W_K(context_embs)  # [B, N, d]
        V = self.W_V(context_embs)  # [B, N, d]

        d = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)  # [B, 1, N]
        attn_weights = self.softmax(scores, dim=-1)  # [B, 1, N]
        attn_out = torch.matmul(attn_weights, V)  # [B, 1, d]

        # attn_out = self.match_l2_norm(attn_out, query_emb)

        refined = query_emb + attn_out.squeeze(1)  # [B, d]
        return refined.squeeze(0) if refined.size(0) == 1 else refined

    def match_l2_norm(self, attn_output, query):
        query_norm = torch.norm(query, p=2, dim=-1, keepdim=True).unsqueeze(1)
        attn_norm = torch.norm(attn_output, p=2, dim=-1, keepdim=True)
        scaled_attn = attn_output * (query_norm / (attn_norm + 1e-8))
        return scaled_attn


    def plot_att_matrix(self, query_emb, context_embs, save_path):
        # Ensure query_emb is [1, d] for matrix ops
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)  # [1, d]
            # context_embs = context_embs.unsqueeze(0)  # [1, N, d]
        elif query_emb.dim() == 2 and context_embs.dim() == 2:
            # Single-query mode but both inputs are 2D
            context_embs = context_embs.unsqueeze(0)  # [1, N, d]
        elif query_emb.dim() != 2 or context_embs.dim() != 3:
            raise ValueError("query_emb must be [d] or [B, d], and context_embs must be [N, d] or [B, N, d]")

        Q = self.W_Q(query_emb)  # [1, d]
        K = self.W_K(context_embs)  # [N, d]
        V = self.W_V(context_embs)

        d = Q.size(-1)
        scores = Q @ K.transpose(0, 1) / (d ** 0.5)  # [1, N]
        attn_weights = self.softmax(scores, dim=-1)  # [1, N]
        attn_out = attn_weights @ V  # [1, d]

        refined = query_emb + attn_out  # [1, d]
        # Convert to NumPy for plotting (detach and move to CPU if on GPU)

        os.makedirs(save_path, exist_ok=True)

        # --- Key input Context Embeddings ---
        ctx_np = context_embs.detach().cpu().numpy()
        plt.figure()
        plt.imshow(ctx_np, aspect='auto')
        plt.colorbar()
        plt.title("Key Inputs (Context Embeddings)")
        plt.xlabel("Embedding Dim")
        plt.ylabel("Context Index")
        plt.savefig(save_path + "/K_input.png")
        plt.close()
        # --- W_K ---
        W_K_np = self.W_K.weight.detach().cpu().numpy()
        plt.figure()
        plt.imshow(W_K_np, aspect='auto')
        plt.colorbar()
        plt.title("W_K Matrix")
        plt.xlabel("Input Dim")
        plt.ylabel("Output Dim")
        plt.savefig(save_path +"/K_weights.png")
        plt.close()
        # --- Plot 5: Projected K ---
        K_np = K.detach().cpu().numpy()
        plt.figure()
        plt.imshow(K_np, aspect='auto')
        plt.colorbar()
        plt.title("Projected Keys (K)")
        plt.xlabel("Embedding Dim")
        plt.ylabel("Context Index")
        plt.savefig(save_path + "/K_proj.png")
        plt.close()

        # --- Query Input ---
        query_emb_np = query_emb.detach().cpu().numpy()
        plt.figure()
        plt.imshow(query_emb_np, aspect='auto')
        plt.colorbar()
        plt.title("Query Input (Original Tag)")
        plt.xlabel("Input Dim")
        plt.savefig(save_path + "/Q_input.png")
        plt.close()
        # --- W_Q ---
        W_Q_np = self.W_Q.weight.detach().cpu().numpy()
        plt.figure()
        plt.imshow(W_Q_np, aspect='auto')
        plt.colorbar()
        plt.title("W_Q Matrix")
        plt.xlabel("Input Dim")
        plt.ylabel("Output Dim")
        plt.savefig(save_path + "/Q_weights.png")
        plt.close()
        # --- Projected Q ---
        Q_np = Q.detach().cpu().numpy()
        plt.figure()
        plt.imshow(Q_np, aspect='auto')
        plt.colorbar()
        plt.title("Projected Query (Q)")
        plt.xlabel("Embedding Dim")
        plt.ylabel("Query Index")
        plt.savefig(save_path + "/Q_proj.png")
        plt.close()

        # --- W_V ---
        W_V_np = self.W_V.weight.detach().cpu().numpy()
        plt.figure()
        plt.imshow(W_V_np, aspect='auto')
        plt.colorbar()
        plt.title("W_V Matrix")
        plt.xlabel("Input Dim")
        plt.ylabel("Output Dim")
        plt.savefig(save_path + "/V_weights.png")
        plt.close()
        # --- Projected values V ---
        V_np = V.detach().cpu().numpy()
        plt.figure()
        plt.imshow(V_np, aspect='auto')
        plt.colorbar()
        plt.title("Projected Values (V)")
        plt.xlabel("Input Dim")
        plt.ylabel("Output Dim")
        plt.savefig(save_path + "/V_proj.png")
        plt.close()

        # --- Attention Weights ---
        attn_weights_np = attn_weights.squeeze(0).detach().cpu().numpy()  # shape [N]
        plt.figure()
        plt.imshow(attn_weights_np[None, :], aspect='auto')
        plt.colorbar()
        plt.title("Attention Weights (Result of Softmax K*V)")
        plt.xlabel("Context Index")
        plt.ylabel("Query Index")
        plt.savefig(save_path + "/attn_weights.png")
        plt.close()
        # --- attention output ---
        attn_out_np = attn_out.detach().cpu().numpy()
        plt.figure()
        plt.imshow(attn_out_np, aspect='auto')
        plt.colorbar()
        plt.title("Attended Values (att * V)")
        plt.savefig(save_path + "/attn_output.png")
        plt.close()

        # --- attention output ---
        refined_query_np = refined.detach().cpu().numpy()
        plt.figure()
        plt.imshow(refined_query_np, aspect='auto')
        plt.colorbar()
        plt.title("Final (att * V + Q_input)")
        plt.savefig(save_path + "/attn_output_plus_residual.png")
        plt.close()



class GaussianEmbedding(nn.Module):
    def __init__(self, d: int = 10, min_val: float = 20.0, max_val: float = 60.0, init_sigma: float = 2.0):
        """
        Args:
            d: size of the embedding space (output vector dimension)
            min_val: minimum value for input normalization (e.g., 20 km)
            max_val: maximum value for input normalization (e.g., 60 km)
            init_sigma: initial value of standard deviation (learnable)
        """
        super().__init__()
        self.d = d
        self.min_val = min_val
        self.max_val = max_val
        sigma = torch.tensor(init_sigma, dtype=torch.float32, requires_grad=True)
        self.sigma = nn.Parameter(sigma)  # Learnable scalar

    def forward(self, x: float or torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: single float or 0-dim tensor (e.g., a length like 37.5)
        Returns:
            A soft Gaussian vector of shape [d]
        """
        if isinstance(x, float) or isinstance(x, int):
            x = torch.tensor(x, dtype=torch.float32, device=self.sigma.device)
        else:
            x = x.to(self.sigma.device)

        # Normalize x ∈ [0, 1]
        x_norm = (x - self.min_val) / (self.max_val - self.min_val)
        x_norm = torch.clamp(x_norm, 0, 1)

        # Compute center in [0, d-1]
        mu = x_norm * (self.d - 1)

        # Embedding positions
        i = torch.arange(self.d, dtype=torch.float32, device=self.sigma.device)

        # Gaussian bump
        gauss = torch.exp(-0.5 * ((i - mu) / self.sigma) ** 2)
        gauss = gauss / gauss.sum()  # normalize

        return gauss  # shape: [d]


