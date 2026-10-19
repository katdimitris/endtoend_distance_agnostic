"""
Convert a transmitter/receiver trained with eakirtas/endtoend_plain (v4_haetae: transmitter.mdl / receiver.mdl state
dicts) into this repo's checkpoint layout, so inference.py, retrain.py and the analysis scripts load it like any
other model: <out>/{transmitter,channel,receiver}.pt + parameters.pt + parameters.json.

Their networks use the same layer names as nets.Transimitter / nets.Receiver (layer_in, layer_1, layer_out; the
activations hold no weights), so the state dicts load as they are. What has to be recorded in parameters.pt:
  - transmitter hidden width 2M = 128 (ours: M = 64)       -> parameters['tx_hidden']
  - their activations (photonic sigmoid everywhere by default)
  - their channel settings: same Channel_v3 ops, here n=40, 320 GSa/s, fc=0.1 (32 GHz), constant sigma, fixed km
  - their batch of 253 messages (23 sequences): the MZM min-max is batch-dependent -> parameters['eval_batch'];
    inference.py uses it unless --batch_size is given

Usage:
  python import_kirtas.py --src DIR_WITH_MDL_FILES --out pretrained/kirtas-25km-0.025/photonic \
      --activations photonic --sigma 0.025 --km 25
"""
import argparse
import json
import os
import sys

import torch as T

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True, help='folder with transmitter.mdl and receiver.mdl')
ap.add_argument('--out', required=True, help='checkpoint folder to create')
ap.add_argument('--activations', choices=['photonic', 'nonphotonic'], default='photonic',
                help="photonic: photonic sigmoid in every Tx/Rx activation (their default); "
                     "nonphotonic: Tx relu,relu,sigmoid / Rx relu,relu (this repo's profile)")
ap.add_argument('--sigma', type=float, required=True, help='constant AWGN std the model was trained with')
ap.add_argument('--km', type=float, required=True, help='fibre length the model was trained at')
ap.add_argument('--n', type=int, default=40)
ap.add_argument('--batch', type=int, default=253, help='their training batch in messages (23 sequences of 11)')
ap.add_argument('--steps', type=int, default=20000, help='optimisation steps the model was trained for (recorded only)')
ap.add_argument('--seed', type=int, default=1)
a = ap.parse_args()

tx_act = 'photonic_sigmoid,photonic_sigmoid,photonic_sigmoid' if a.activations == 'photonic' else 'relu,relu,sigmoid'
rx_act = 'photonic_sigmoid,photonic_sigmoid' if a.activations == 'photonic' else 'relu,relu'
km = int(a.km) if float(a.km).is_integer() else a.km
sys.argv = ['train.py', '--n', str(a.n), '--bits', '6', '--seed', str(a.seed), '--batch_size', str(a.batch),
            '--lr', '0.0005', '--train_iter', str(a.steps), '--fixed_sigma', str(a.sigma),
            '--train_lengths', f'{km},{km}', '--transfer_lengths', f'{km},{km}',
            '--receiver_type', 'baseline', '--transmitter_type', 'baseline',
            '--transmitter_activations', tx_act, '--receiver_activations', rx_act]
from train import parser, build_parameters   # noqa: E402  (after sys.argv is set)
from parameters import get_models            # noqa: E402

parameters = build_parameters(parser.parse_args())
parameters['device'] = T.device('cpu')
parameters['tx_hidden'] = 2 * parameters['M']
parameters['eval_batch'] = a.batch
parameters['source'] = {'repo': 'https://github.com/eakirtas/endtoend_plain', 'commit': 'da223e5',
                        'channel': 'v4_haetae Channel_v4 (same ops as channel.Channel_v3)',
                        'training': f'{a.steps} steps = 1000 epochs x {a.steps // 1000} batches of {a.batch} messages, '
                                    'step lr 5e-4 -> 2e-7 every 100 epochs, Adam re-created every epoch',
                        'imported_from': os.path.abspath(a.src)}

models = get_models(parameters)
for name in ('transmitter', 'receiver'):
    state = T.load(os.path.join(a.src, f'{name}.mdl'), map_location='cpu', weights_only=True)
    models[name].load_state_dict(state)          # strict: fails loudly on any shape / name mismatch

os.makedirs(a.out, exist_ok=True)
for name, model in models.items():
    T.save(model.state_dict(), os.path.join(a.out, f'{name}.pt'))
T.save(parameters, os.path.join(a.out, 'parameters.pt'))
with open(os.path.join(a.out, 'parameters.json'), 'w') as f:
    json.dump({k: str(v) for k, v in parameters.items()}, f, indent=2, sort_keys=True)
print(f'wrote {a.out}: tx {tx_act} (hidden {parameters["tx_hidden"]}), rx {rx_act}, sigma {a.sigma}, {km} km, '
      f'eval batch {a.batch}')
