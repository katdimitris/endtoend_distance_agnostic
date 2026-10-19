import numpy as np
import torch as T
from utils import PhotonicSigmoid, ClippingReLU
from channel import Channel_v3
from nets import Receiver, Transimitter, FiLMTransimitter, FiLMReceiver, EqualizerReceiver


def get_channel(parameters):
    channel = Channel_v3(
        parameters['sampl_num'],
        parameters['fsym_length'],
        parameters['band_fact'],
        parameters['fc'],
        parameters['std_km_table'],
        parameters['snr'],
        parameters['ER'],
        parameters['sample_rate'],
        parameters['train_lengths'],
        parameters['length_std'],
        parameters['d_ps_nm_km'],
        parameters['reference_wavelength_nm'],
        2.9979e5,
        parameters['N'],
        device=parameters['device'],
        full_output=parameters.get('full_eval', False),
        sliding=parameters.get('sliding_eval', False),
        window_margin=parameters.get('window_margin', None),
        # checkpoints saved before std_interp existed were trained with the single LSQ fit
        std_interp=parameters.get('std_interp', 'loglinear_fit'),
    )
    return channel

def get_transmitter(parameters):
    n_tags = 0
    # .get() defaults keep old checkpoints (saved before --transmitter_activations existed)
    # loadable: they get today's previous hardcoded behavior (relu, relu, sigmoid).
    tx_activations = parameters.get('transmitter_activations', ('relu', 'relu', 'sigmoid'))
    # hidden width defaults to M (64); checkpoints imported from eakirtas/endtoend_plain set tx_hidden=2M (128)
    tx_hidden = parameters.get('tx_hidden', parameters['M'])
    if parameters['transmitter_type'] == 'baseline':
        transmitter = Transimitter(n_input=parameters['M']+n_tags, hidden=tx_hidden, n_output=parameters['n'],
                                   activations=tx_activations, parameters=parameters)
    elif parameters['transmitter_type'] == 'film':
        transmitter = FiLMTransimitter(n_input=parameters['M']+n_tags, hidden=parameters['M'], n_output=parameters['n'],
                                       min_length=parameters['train_lengths'][0],
                                       max_length=parameters['train_lengths'][1],
                                       film_layers=parameters['film_layers'],
                                       length_encoding=parameters['length_encoding'],
                                       position=parameters['position'], film_mode=parameters['film_mode'],
                                       film_trunk_layers=parameters['mlp_layers'],
                                       activations=tx_activations, parameters=parameters,
                                       )
    return transmitter


def get_receiver(parameters):
    n_tags = 0
    # .get() default keeps old checkpoints (saved before --receiver_activations existed) loadable.
    rx_activations = parameters.get('receiver_activations', ('relu', 'relu'))
    if parameters['receiver_type'] == 'baseline':
        receiver = Receiver(n_input=parameters['n'] + n_tags, hidden=2 * parameters['M'], n_output=parameters['M'],
                            activations=rx_activations, parameters=parameters)
    elif parameters['receiver_type'] == 'film':
        receiver = FiLMReceiver(n_input=parameters['n'] + n_tags, hidden=2 * parameters['M'], n_output=parameters['M'],
                                min_length=parameters['train_lengths'][0], max_length=parameters['train_lengths'][1],
                                film_layers=parameters['film_layers'], length_encoding=parameters['length_encoding'],
                                position=parameters['position'], film_mode=parameters['film_mode'], film_trunk_layers=parameters['mlp_layers'],
                                activations=rx_activations, parameters=parameters)
    elif parameters['receiver_type'] == 'equalizer':
        margin = parameters.get('window_margin', 2)
        # input is the WindowDeserializer's (2*margin+1) symbols wide; each symbol is n samples
        n_in = (2 * margin + 1) * parameters['n']
        receiver = EqualizerReceiver(n_input=n_in, hidden=2 * parameters['M'], n_output=parameters['M'],
                                     min_length=parameters['train_lengths'][0],
                                     max_length=parameters['train_lengths'][1],
                                     length_encoding=parameters['length_encoding'],
                                     rank=parameters.get('rank', 8))
    elif parameters['receiver_type'] == 'wide':
        # ablation: wider neighbor-window context but NO length conditioning at all -- a plain
        # unconditioned MLP receiver on the WindowDeserializer's wide input. Isolates whether the
        # extra ISI context alone helps, separate from the length-conditioned mixing.
        margin = parameters.get('window_margin', 2)
        n_in = (2 * margin + 1) * parameters['n']
        receiver = Receiver(n_input=n_in, hidden=2 * parameters['M'], n_output=parameters['M'])
    else:
        raise ValueError(f"Unsupported receiver_type: {parameters['receiver_type']!r} "
                          f"(expected 'baseline', 'film', 'equalizer', or 'wide')")

    return receiver

def get_channel_parameters(initializer=None):
    parameters = {}
    parameters['fsym_length'] = 32
    parameters[
        'band_fact'] = 0.01  # Inside LPF_Sync will determine the number of taps and so the filter sharpness
    parameters['snr'] = 38  #   SNR = ENOB*6.02 + 1.76
    parameters['sample_rate'] = 320e9 * 1.0 # 336e9 * 1.0
    parameters['fc'] = (
        (32e9) / (parameters['sample_rate'])
    )  # Cutoff frequency as a fraction of the sampling rate
    # AWGN std, calibrated per fiber length; interpolated/extrapolated at forward time based on
    # the current length (see AWGN_STD_v3 in channel.py)
    # Previous table (used by every run up to range_10_60_n40 / range_20_60_n40 and by
    # pretrained/10-60km, pretrained/20-60km),
    # with the single least-squares log-linear fit ('loglinear_fit'):
    # parameters['std_km_table'] = (
    #     (20, 0.0248), (30, 0.03126), (40, 0.03936), (50, 0.04955),
    #     (60, 0.06238), (70, 0.07853), (80, 0.09886),
    # )
    parameters['std_km_table'] = (
        (25, 0.0273), (29, 0.0329), (36, 0.0476), (40, 0.0554), (47, 0.1058),
    )
    # exact through every measured point, extended by the end segments outside 25-47km
    parameters['std_interp'] = 'piecewise_loglinear'
    parameters['ER'] = 20
    parameters['d_ps_nm_km'] = 17.0
    parameters['reference_wavelength_nm'] = 1550.0
    parameters['sampl_num'] = 4
    parameters['enob'] = 6
    parameters['p'] = 1

    parameters['M'] = 64  # Input length one-hot-vector 64
    parameters['n'] = 40  # Transmitter output 48

    # Input of transmitter (11, 64) -> (batch size, one-hot-vector)
    # Output of transmitter (11, 48) -> (batch size, signal)
    # (253, 48) -> (23, 48 * 11)
    parameters['N'] = 11  # Transmitted bits over N*n ++++++++
    parameters['initializer'] = initializer
    parameters['length_std'] = 0.0

    return parameters

def get_models(parameters):
    transmitter = get_transmitter(parameters).to(parameters['device'])
    channel = get_channel(parameters).to(parameters['device'])
    receiver = get_receiver(parameters).to(parameters['device'])

    models = {
        'receiver': receiver,
        'channel': channel,
        'transmitter': transmitter
    }

    return models