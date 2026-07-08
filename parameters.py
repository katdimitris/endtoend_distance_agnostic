import numpy as np
import torch as T
from utils import PhotonicSigmoid, ClippingReLU
from channel import Channel_v3
from nets import Receiver, Transimitter, FiLMTransimitter, FiLMReceiver


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
        device=parameters['device']
    )
    return channel

def get_transmitter(parameters):
    n_tags = 0
    if parameters['transmitter_type'] == 'baseline':
        transmitter = Transimitter(n_input=parameters['M']+n_tags, hidden=parameters['M'], n_output=parameters['n'])
    elif parameters['transmitter_type'] == 'film':
        transmitter = FiLMTransimitter(n_input=parameters['M']+n_tags, hidden=parameters['M'], n_output=parameters['n'],
                                       min_length=parameters['train_lengths'][0],
                                       max_length=parameters['train_lengths'][1],
                                       film_layers=parameters['film_layers'],
                                       length_encoding=parameters['length_encoding'],
                                       position=parameters['position'], film_mode=parameters['film_mode'],
                                       film_trunk_layers=parameters['mlp_layers']
                                       )
    return transmitter


def get_receiver(parameters):
    n_tags = 0
    if parameters['receiver_type'] == 'baseline':
        receiver = Receiver(n_input=parameters['n'] + n_tags, hidden=2 * parameters['M'], n_output=parameters['M'])
    elif parameters['receiver_type'] == 'film':
        receiver = FiLMReceiver(n_input=parameters['n'] + n_tags, hidden=2 * parameters['M'], n_output=parameters['M'],
                                min_length=parameters['train_lengths'][0], max_length=parameters['train_lengths'][1],
                                film_layers=parameters['film_layers'], length_encoding=parameters['length_encoding'],
                                position=parameters['position'], film_mode=parameters['film_mode'], film_trunk_layers=parameters['mlp_layers'])
    else:
        raise ValueError(f"Unsupported receiver_type: {parameters['receiver_type']!r} "
                          f"(expected 'baseline' or 'film')")

    return receiver

def get_channel_parameters(initializer=None):
    parameters = {}
    parameters['fsym_length'] = 32
    parameters[
        'band_fact'] = 0.01  # Inside LPF_Sync will determine the number of taps and so the filter sharpness
    parameters['snr'] = 38  #   SNR = ENOB*6.02 + 1.76
    parameters['sample_rate'] = 336e9 * 1.0
    parameters['fc'] = (
        (32e9) / (parameters['sample_rate'])
    )  # Cutoff frequency as a fraction of the sampling rate
    # AWGN std, calibrated per fiber length; interpolated/extrapolated at forward time based on
    # the current length (see AWGN_STD_v3 in channel.py)
    parameters['std_km_table'] = (
        (20, 0.0248), (30, 0.03126), (40, 0.03936), (50, 0.04955),
        (60, 0.06238), (70, 0.07853), (80, 0.09886),
    )
    parameters['ER'] = 20
    parameters['d_ps_nm_km'] = 17.0
    parameters['reference_wavelength_nm'] = 1550.0
    parameters['sampl_num'] = 4
    parameters['enob'] = 6
    parameters['p'] = 1

    parameters['M'] = 64  # Input length one-hot-vector 64
    parameters['n'] = 48  # Transmitter output 48

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