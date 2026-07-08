
import os

import numpy as np
import torch as T
from tqdm import tqdm

from train import parser, build_parameters, build_exp_identifiers
from parameters import get_models
from utils import get_random_ds, ErrorCounter, OneHotToInt, AverageMeter, fix_seed

parser.add_argument("--eval_kms", type=lambda s: [int(x) for x in s.split(',')], required=True,
                     help="comma-separated km values to evaluate, e.g. '0,10,17,25,35,42'")
parser.add_argument("--checkpoint_dir", type=str, default=None,
                     help="explicit path to the checkpoint folder (containing transmitter.pt/"
                          "channel.pt/receiver.pt/parameters.pt). If omitted, it's located from "
                          "these same CLI flags using train.py's naming convention.")
parser.add_argument("--output_csv", type=str, default=None,
                     help="if set, also write km,bler,ce_loss rows to this CSV path.")


def load_models(parameters, checkpoint_dir):
    models = get_models(parameters)
    device = parameters['device']
    for name, model in models.items():
        state = T.load(f'{checkpoint_dir}/{name}.pt', map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
    return models


def evaluate(models, parameters, loader, length):
    transmitter, channel, receiver = models['transmitter'], models['channel'], models['receiver']
    device = parameters['device']

    criterion = T.nn.CrossEntropyLoss()
    error_counter = ErrorCounter(parameters['bits'])
    onehot_to_int = OneHotToInt()

    channel.set_length(length=length)
    loss_meter = AverageMeter()
    errors, bits_counter = 0, 0
    with T.no_grad():
        for inputs, targets in tqdm(loader, desc=f"km={length}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            t_outputs = transmitter(inputs, length)
            c_outputs = channel(t_outputs, length)
            r_outputs = receiver(c_outputs, length)

            loss = criterion(r_outputs, targets)
            loss_meter.update(loss.item(), inputs.size(0))
            errors_t, bits = error_counter(onehot_to_int(r_outputs.detach()), targets)
            errors += errors_t
            bits_counter += bits

    bler = errors.astype(np.float64) / float(bits_counter)
    return bler, loss_meter.avg


def main():
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        # only used to locate the checkpoint dir -- the architecture itself comes from
        # that checkpoint's own saved parameters.pt, not from these CLI flags.
        cli_parameters = build_parameters(args)
        exp_dir, exp_name = build_exp_identifiers(args, cli_parameters)
        checkpoint_dir = f'checkpoints/{exp_dir}{exp_name}'

    print(f"Loading checkpoint from {checkpoint_dir}")
    parameters = T.load(os.path.join(checkpoint_dir, 'parameters.pt'), weights_only=False)
    parameters['device'] = T.device(f"cuda:{args.gpu_id}" if T.cuda.is_available() else "cpu")
    parameters['val_size'] = args.val_size
    parameters['batch'] = args.batch_size

    models = load_models(parameters, checkpoint_dir)

    loader = get_random_ds(size=parameters['val_size'], max_num=parameters['M'],
                            N_symbols=parameters['N'], device=parameters['device'],
                            batch_size=parameters['batch'], dtype=T.float)

    rows = []
    print(f"{'km':>6}  {'BLER':>12}  {'CE Loss':>12}")
    pbar = tqdm(args.eval_kms, desc="Evaluating")
    for km in pbar:
        pbar.set_postfix(km=km)
        bler, loss = evaluate(models, parameters, loader, km)
        print(f"{km:>6}  {bler:>12.6e}  {loss:>12.6f}")
        rows.append((km, bler, loss))

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
        with open(args.output_csv, 'w') as f:
            f.write("km,bler,ce_loss\n")
            for km, bler, loss in rows:
                f.write(f"{km},{bler},{loss}\n")
        print(f"Wrote {args.output_csv}")


if __name__ == '__main__':
    args = parser.parse_args()
    fix_seed(args.seed)
    main()
