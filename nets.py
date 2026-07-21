import torch
import torch as T
import torch.nn as nn
import math

from utils import PhotonicSigmoid, ClippingReLU

ALLOWED_ACTIVATIONS = ('relu', 'sigmoid', 'photonic_sigmoid', 'cliprelu')


def make_activation(name, parameters=None):
    """Builds an activation module by name. 'cliprelu' needs parameters['enob']/['p'] to compute
    its clip point (s_q, the ADC/DAC quantization step) -- same formula used historically for
    this receiver/transmitter pair (s_q = 3*p*10**(-(6.02*enob+1.76)/10)); falls back to enob=6,
    p=1 (this repo's defaults) if parameters is None."""
    name = str(name).lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'photonic_sigmoid':
        return PhotonicSigmoid()
    elif name == 'cliprelu':
        p = parameters.get('p', 1) if parameters else 1
        enob = parameters.get('enob', 6) if parameters else 6
        s_q = 3 * p * 10 ** (-(6.02 * enob + 1.76) / 10.0)
        return ClippingReLU(s_q)
    else:
        raise ValueError(f"Unsupported activation {name!r}; expected one of {ALLOWED_ACTIVATIONS}")


class Receiver(nn.Module):
    def __init__(self, n_input=48, hidden=128, n_output=64, activations=('relu', 'relu'), parameters=None):
        super().__init__()

        self.layer_in = nn.Linear(n_input, hidden)
        self.layer_1 = nn.Linear(hidden, hidden)
        self.layer_out = nn.Linear(hidden, n_output)
        self.act1 = make_activation(activations[0], parameters)
        self.act2 = make_activation(activations[1], parameters)

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
    def __init__(self, n_input=64, hidden=128, n_output=48, activations=('relu', 'relu', 'sigmoid'), parameters=None):
        super().__init__()

        self.layer_in = nn.Linear(n_input, hidden)
        self.layer_1 = nn.Linear(hidden, hidden)
        self.layer_out = nn.Linear(hidden, n_output)
        self.act1 = make_activation(activations[0], parameters)
        self.act2 = make_activation(activations[1], parameters)
        self.act_out = make_activation(activations[2], parameters)

    def forward(self, x, lengths):
        x = self.layer_in(x)
        x = self.act1(x)
        x = self.layer_1(x)
        x = self.act2(x)
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
        activations=('relu', 'relu', 'sigmoid'),
        parameters=None,
    ):
        """
        Same FiLM interface as FiLMReceiver, but keeps the original Transimitter
        backbone dimensions: 64 -> 128 -> 128 -> 48.
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
        self.act1 = make_activation(activations[0], parameters)
        self.act2 = make_activation(activations[1], parameters)
        self.act_out = make_activation(activations[2], parameters)


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
        activations=('relu', 'relu'),
        parameters=None,
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
        self.act1 = make_activation(activations[0], parameters)
        self.act2 = make_activation(activations[1], parameters)

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


class EqualizerReceiver(nn.Module):
    """Length-agnostic receiver targeting what the distortion diagnosis showed actually matters:
    chromatic-dispersion ISI is a length-dependent *convolution* (cross-sample mixing), not a
    per-feature rescale/shift -- so FiLM's beta/gamma condition the wrong axis. This does two
    things FiLM cannot:

      (1) Length-conditioned MIXING: the first layer weight is W(L) = W0 + U diag(alpha(L)) V, a
          low-rank correction whose coefficients alpha(L) come from a hypernetwork on the one-hot
          length. This is a per-length linear mixing across input samples (a length-adapted
          equalizing filter), which the diagnosis found recovers the clean representation to
          R^2 >= 0.976 at every length, vs a per-length diagonal (FiLM) collapsing toward 0.64.
      (2) Wider input context: paired with channel.WindowDeserializer, x already contains the
          neighbor symbols dispersion leaked energy into, so the mixing has that energy to work
          with (the center-only receiver never sees it).

    Zero-init: the hypernetwork's output layers start at zero, so alpha(L)=0 and the per-length
    bias=0 at step 0 -> the model is exactly a plain (unconditioned) wide-context receiver, and can
    only *add* conditioning as it helps. A per-length hidden bias alone reproduces FiLM-beta, so
    this receiver's hypothesis class contains beta-only FiLM: it cannot represent a worse function.
    """
    def __init__(self, n_input, hidden, n_output, min_length, max_length,
                 length_encoding='one_hot', length_step=2.0, rank=8, hyper_hidden=None):
        super().__init__()
        self.n_input = n_input
        self.hidden = hidden
        self.minL = float(min_length)
        self.maxL = float(max_length)
        self.length_step = float(length_step)
        self.length_encoding = str(length_encoding).lower()
        if self.length_encoding != 'one_hot':
            raise ValueError("EqualizerReceiver currently supports length_encoding='one_hot' only, "
                             f"got {length_encoding!r}")
        if self.length_step <= 0:
            raise ValueError("length_step must be > 0")
        self.num_bins = int(math.floor((self.maxL - self.minL) / self.length_step)) + 1
        self.rank = int(rank)

        # backbone (same shape as the baseline Receiver, but a wider input from the window)
        self.layer_in = nn.Linear(n_input, hidden)
        self.layer_1 = nn.Linear(hidden, hidden)
        self.layer_out = nn.Linear(hidden, n_output)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        # low-rank basis for the length-conditioned weight correction W(L)=W0 + U diag(alpha) V
        self.U = nn.Parameter(torch.empty(hidden, self.rank))
        self.V = nn.Parameter(torch.empty(self.rank, n_input))
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

        # hypernetwork: one-hot length -> rank coefficients alpha(L), and -> per-length hidden bias
        hh = hidden if hyper_hidden is None else int(hyper_hidden)
        self.hyper = nn.Sequential(nn.Linear(self.num_bins, hh), nn.ReLU(), nn.Linear(hh, self.rank))
        self.bias_hyper = nn.Linear(self.num_bins, hidden)
        # zero-init the OUTPUT layers so the correction and bias are exactly 0 at init
        nn.init.zeros_(self.hyper[-1].weight)
        nn.init.zeros_(self.hyper[-1].bias)
        nn.init.zeros_(self.bias_hyper.weight)
        nn.init.zeros_(self.bias_hyper.bias)

    def _one_hot(self, length, B, device, dtype):
        if not torch.is_tensor(length):
            length = torch.tensor(length, device=device, dtype=dtype)
        else:
            length = length.to(device=device, dtype=dtype)
        if length.dim() == 0:
            length = length.unsqueeze(0).expand(B)
        elif length.dim() == 1 and length.numel() == 1:
            length = length.expand(B)
        length = length.clamp(self.minL, self.maxL)
        idx = torch.round((length - self.minL) / self.length_step).long().clamp(0, self.num_bins - 1)
        one_hot = torch.zeros(B, self.num_bins, device=device, dtype=dtype)
        one_hot.scatter_(1, idx.view(B, 1), 1.0)
        return one_hot

    def forward(self, x, length):
        B = x.size(0)
        one_hot = self._one_hot(length, B, x.device, x.dtype)
        alpha = self.hyper(one_hot)                    # (B, rank), zero at init
        Vx = x @ self.V.t()                            # (B, rank)
        correction = (alpha * Vx) @ self.U.t()         # (B, hidden) == U diag(alpha) V x, per-sample
        beta = self.bias_hyper(one_hot)                # (B, hidden), zero at init (== FiLM-beta)
        h = self.layer_in(x) + correction + beta
        h = self.act1(h)
        h = self.layer_1(h)
        h = self.act2(h)
        return self.layer_out(h)


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


