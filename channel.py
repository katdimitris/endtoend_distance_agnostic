import torch as T
import torch.nn.functional as F
import math
import numpy as np
from numpy.compat import integer_types

from scipy.fftpack import fftfreq

from utils import signal_normalize, signal_normalize_min_max
from utils import ifnot_create, roll


''' CHANNEL'''

class Channel_v3(T.nn.Module):
    def __init__(
            self,
            sampl_num: float = 16,
            fsym_length: float = 101,
            band_fact: float = 0.121,
            fc: float = 0.095,
            std_km_table=((20, 0.0248), (30, 0.03126), (40, 0.03936), (50, 0.04955),
                          (60, 0.06238), (70, 0.07853), (80, 0.09886)),
            snr: float = 37.88,
            ER: float = 11.37,
            sample_rate: float = 160e9,  # 10GS
            length_km: float = (10.57, 10.57),
            length_std: float = 0.0,
            d_ps_nm_km: float = 16.1,
            reference_wavelength_nm: float = 1550,
            c: float = 3e5,
            N: int = 50 * 127,  #11      Transmitted symbols over the channel
            dtype=T.float,
            device='cuda:0'):  #dtype=T.double

        super().__init__()

        self.parameters = {}
        self.parameters['sampl_num'] = sampl_num
        self.parameters['fsym_length'] = fsym_length
        self.parameters['band_fact'] = band_fact
        self.parameters['std_km_table'] = std_km_table
        self.parameters['snr'] = snr
        self.parameters['ER'] = ER
        self.parameters['sample_rate'] = sample_rate
        self.parameters['fc'] = fc
        self.parameters['length_km'] = length_km
        self.parameters['d_ps_nm_km'] = d_ps_nm_km
        self.parameters['reference_wavelength_nm'] = reference_wavelength_nm
        self.parameters['c'] = c
        self.parameters['N'] = N
        self.parameters['train_length_std'] = length_std
        self.dtype = dtype
        self.device = device

        self.serializer = Serializer(N)
        self.deserializer = Deserializer(N)

        self.lpf = LPFv3_Sync(fc, band_fact, normalize=True, dtype=dtype)
        self.mzm = MZMv3(ER, normalize=True, dtype=dtype)
        self.awgn_std = AWGN_STD_v3(std_km_table, normalize=True, dtype=dtype)
        self.awgn_snr = AWGN_SNR_v3(snr, normalize=True, dtype=dtype)
        self.dispersion = Dispersion_v3(sample_rate, d_ps_nm_km,
                                        reference_wavelength_nm, length_km,
                                        length_std, c, True, dtype, self.device)

    def forward(self, x, length=None):
        x = self.serializer(x)
        x = self.lpf(x)  # Filtering
        x = roll(x, 1)
        x = self.awgn_snr(x)  # Quantization Noise DAC
        x = self.mzm(x)
        x = self.dispersion(x)
        x = self.awgn_std(x)
        x = self.lpf(x)  # Filtering
        x = roll(x, 1)
        x = self.awgn_snr(x)  # Quantization Noise ADC
        x = self.deserializer(x)
        return x

    def get_length(self):
        return self.dispersion.get_length()

    def set_length(self, length):
        self.dispersion.set_length(length)
        self.awgn_std.set_length(length)


''' CHANNEL COMPONENTS'''
PI = T.tensor(np.pi, requires_grad=False)
EPSILON = T.tensor(np.e, requires_grad=False)

class Dispersion_v3(T.nn.Module):
    def __init__(
            self,
            sample_rate=50e9,
            d_ps_nm_km=16.7,
            reference_wavelength_nm=1550,
            length_km=(10,10),
            length_std=0.0,
            c=3e5,  # speed of light nm/ps
            normalize=False,
            dtype=T.double,
            device='cuda:0'):

        super().__init__()
        self.sample_rate = sample_rate
        self.dps_nm_km = d_ps_nm_km
        self.ref_wv = reference_wavelength_nm
        self.length_km = length_km
        self.length_std = length_std
        self.normalize = normalize
        self.c = c
        self.D = None
        self.dtype = dtype
        self.device = device

    def set_length(self, length):
        self.cur_length = length

    def set_length_std(self, std):
        self.length_std = std

    def get_length(self):
        return self.cur_length

    def get_d(self, x):
        freq = 1e-12 * T.fft.fftfreq(
            x.size()[1], 1.0 / self.sample_rate, device=self.device)
        omega = 2.0 * PI * freq
        omega2 = omega ** 2
        beta = -self.dps_nm_km * (self.ref_wv ** 2.0) / (2.0 * PI * self.c)

        if T.is_tensor(self.cur_length) and self.cur_length.dim() > 0:
            phase = 0.5 * beta * self.cur_length.view(-1, 1) * omega2.view(1, -1)  # (bs, d)
            self.D = T.exp(1j * phase)
        else:
            self.D = T.exp(1j / 2.0 * beta * self.cur_length * omega2).expand_as(x)

        return self.D

    def __call__(self, x):
        D = self.get_d(x)
        x = T.fft.fft2(x, dim=1)
        x = x * D
        x = T.fft.ifft(x, dim=1)
        x = T.pow(x.real, 2)

        return x

class MZMv3(T.nn.Module):
    def __init__(self, ER=11.7, normalize=True, dtype=T.double):
        super().__init__()
        self.normalize = normalize
        self.ER = ER
        self.pi = PI.type(dtype)

    def forward(self, x: T.Tensor):
        ER_ratio = 1 / (10 ** (self.ER / 10))

        x = T.sin(x * self.pi / 4)

        return signal_normalize_min_max(x, ER_ratio, 1,
                                        1) if self.normalize else x



class AWGN_STD_v3(T.nn.Module):
    def __init__(self, std_km_table, normalize=False, dtype=T.double):
        super().__init__()
        self.normalize = normalize
        self.dtype = dtype
        self.cur_length = None

        # AWGN std grows with fiber length (ASE-noise-style). Fit log10(std) = a + b*km
        # to the calibrated (km, std) table so it can be evaluated at any length, not just
        # the tabulated ones -- same idea as Dispersion_v3 evaluating its phase from length.
        kms, stds = zip(*std_km_table)
        b, a = np.polyfit(np.array(kms, dtype=np.float64), np.log10(np.array(stds, dtype=np.float64)), 1)
        self.log_std_a = float(a)
        self.log_std_b = float(b)

    def set_length(self, length):
        self.cur_length = length

    def get_length(self):
        return self.cur_length

    def get_std(self):
        return 10.0 ** (self.log_std_a + self.log_std_b * self.cur_length)

    def forward(self, x):
        std = self.get_std()
        noise = T.randn(x.size(), device=x.device, dtype=x.dtype)
        if T.is_tensor(std) and std.dim() > 0:
            noise = noise * std.view(-1, 1)
        else:
            noise = noise * std
        puls_norm_gauss = T.add(x, noise).to(x.device)
        return puls_norm_gauss


class AWGN_SNR_v3(T.nn.Module):
    def __init__(self, snr, normalize=False, dtype=T.double):
        super().__init__()
        self.snr = snr
        self.normalize = normalize
        self.dtype = dtype
        self.n_0 = T.tensor(10 ** (-self.snr / 10),
                            dtype=dtype,
                            requires_grad=False)

    def forward(self, x):
        sig_noise = T.sqrt(self.n_0) * T.randn(x.size()).type(self.dtype).to(
            x.device)
        puls_norm_gauss = T.add(x, sig_noise)

        return puls_norm_gauss




class Serializer(T.nn.Module):
    def __init__(self, N_symbols=11):
        super(Serializer, self).__init__()
        self.N_symbols = N_symbols

    def __call__(self, x):
        return x.view(-1, x.size()[1] * self.N_symbols)


class ExperimentalSerializer(T.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x


class Deserializer(T.nn.Module):
    def __init__(
            self,
            N_symbols=11,
    ):
        super(Deserializer, self).__init__()
        self.N_symbols = N_symbols

    def __call__(self, x):
        n_signal = x.size()[1] // self.N_symbols

        x = x[:,
            int(self.N_symbols / 2) * n_signal:int(self.N_symbols / 2 + 1) *
                                               n_signal]

        return x.view(-1, n_signal)


class ExperimentalDeserializer(T.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x[:, 1:-1]


PI = T.tensor(np.pi, dtype=T.float64, requires_grad=False)
EPSILON = T.tensor(np.e, dtype=T.float64, requires_grad=False)


''' FILTERS COMPONENTS '''
class LPFv3_Sync(T.nn.Module):
    def __init__(self, fc, band_fact, normalize=False, dtype=T.float):
        super().__init__()
        self.normalize = normalize

        N = int(np.ceil((4 / band_fact)))
        n = np.arange(N)

        H = np.sinc(2 * fc * (n - (N - 1) / 2))
        w = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
            0.08 * np.cos(4 * np.pi * n / (N - 1))

        hw = H * w
        hw = hw / np.sum(hw)

        self.h = T.tensor(hw).float()

        self.kernel = self.h.expand((1, 1, -1))
        self.padding = math.ceil(((self.kernel.size()[2] - 1)) / 2)

    def forward(self, x):
        batch_size = x.size()[0]
        input_x = x.view(batch_size, 1, -1)
        if (self.kernel.device != x.device):
            self.kernel = self.kernel.to(x.device)

        convovled = F.conv1d(input_x,
                             self.kernel,
                             padding=self.padding,
                             groups=1)
        x = convovled.view(batch_size, -1)

        return x


class LPFv3(T.nn.Module):
    def __init__(self,
                 fsym_length,
                 sampl_num,
                 band_fact,
                 normalize=False,
                 dtype=T.double):
        super().__init__()
        self.normalize = normalize

        ff = T.linspace(-fsym_length / 2,
                        fsym_length / 2,
                        sampl_num * fsym_length + 1,
                        dtype=T.double,
                        requires_grad=False)

        self.h = T.sqrt(PI) / band_fact * T.exp(-(PI * ff /
                                                  band_fact)**2).type(dtype)
        self.kernel = self.h.expand((1, 1, -1))
        self.padding = math.ceil(((self.kernel.size()[2] - 1)) / 2)

    def forward(self, x):
        batch_size = x.size()[0]
        input_x = x.view(batch_size, 1, -1)
        if (self.kernel.device != x.device):
            self.kernel = self.kernel.to(x.device)

        convovled = F.conv1d(input_x,
                             self.kernel,
                             padding=self.padding,
                             groups=1)
        x = convovled.view(batch_size, -1)

        return signal_normalize(x, 1) if self.normalize else x