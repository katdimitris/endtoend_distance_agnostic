
import os

import numpy as np
import torch as T
from tqdm import tqdm

from train import parser, build_parameters, build_exp_identifiers
from parameters import get_models
from utils import get_random_ds, ErrorCounter, OneHotToInt, AverageMeter, fix_seed, OneHotVecGenerator

parser.add_argument("--eval_kms", type=lambda s: [int(x) for x in s.split(',')], required=True,
                     help="comma-separated km values to evaluate, e.g. '0,10,17,25,35,42'")
parser.add_argument("--checkpoint_dir", type=str, default=None,
                     help="explicit path to the checkpoint folder (containing transmitter.pt/"
                          "channel.pt/receiver.pt/parameters.pt). If omitted, it's located from "
                          "these same CLI flags using train.py's naming convention.")
parser.add_argument("--output_csv", type=str, default=None,
                     help="if set, also write km,bler,ce_loss rows to this CSV path.")
parser.add_argument("--eval_mode", type=str, default="center", choices=["center", "full", "sliding"],
                     help="'center' (default): decode only the middle symbol of each "
                          "N_symbols-block, matching how the model was trained. 'full': decode "
                          "every symbol in each block (via FullDeserializer + per-position "
                          "targets) instead of discarding the other N_symbols-1 -- each symbol "
                          "still gets real ISI from its neighbors during channel simulation, "
                          "this only changes what gets evaluated afterward. 'sliding': like "
                          "'full' but windows are built with stride 1 over one continuous "
                          "stream (SlidingSerializer) instead of disjoint N_symbols blocks, so "
                          "every decoded symbol gets its TRUE neighbors instead of a "
                          "zero-padded fake edge near a block boundary -- only the first/last "
                          "N_symbols//2 symbols of the whole stream are excluded. Same "
                          "checkpoint works for all three; only the transmitter/receiver "
                          "weights matter.")


def load_models(parameters, checkpoint_dir):
    models = get_models(parameters)
    device = parameters['device']
    for name, model in models.items():
        state = T.load(f'{checkpoint_dir}/{name}.pt', map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
    return models


def evaluate(models, parameters, loader, length, track_positions=False):
    transmitter, channel, receiver = models['transmitter'], models['channel'], models['receiver']
    device = parameters['device']
    N = parameters['N']

    criterion = T.nn.CrossEntropyLoss()
    error_counter = ErrorCounter(parameters['bits'])
    onehot_to_int = OneHotToInt()

    channel.set_length(length=length)
    loss_meter = AverageMeter()
    errors, bits_counter, n_samples = 0, 0, 0
    pos_errors = np.zeros(N, dtype=np.float64) if track_positions else None
    pos_bits = np.zeros(N, dtype=np.float64) if track_positions else None

    with T.no_grad():
        for inputs, targets in tqdm(loader, desc=f"km={length}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            t_outputs = transmitter(inputs, length)
            c_outputs = channel(t_outputs, length)
            r_outputs = receiver(c_outputs, length)

            loss = criterion(r_outputs, targets)
            loss_meter.update(loss.item(), inputs.size(0))

            preds = onehot_to_int(r_outputs.detach())
            errors_t, bits = error_counter(preds, targets)
            errors += errors_t
            bits_counter += bits
            n_samples += targets.size(0)

            if track_positions:
                # preds/targets are ordered [group0_pos0..N-1, group1_pos0..N-1, ...] (no
                # shuffling in the loader), so every N-th element starting at p is position p.
                for p in range(N):
                    e, b = error_counter(preds[p::N], targets[p::N])
                    pos_errors[p] += e
                    pos_bits[p] += b

    bler = errors.astype(np.float64) / float(bits_counter)
    details = {"n_samples": n_samples, "n_bits": bits_counter, "n_errors": int(errors)}
    if track_positions:
        details["per_position_bler"] = (pos_errors / pos_bits).tolist()
    return bler, loss_meter.avg, details


def evaluate_sliding(models, parameters, length, val_size, chunk_size):
    """Like evaluate(..., track_positions=True) but every decoded symbol gets its true stream
    neighbors (stride-1 windows via SlidingSerializer) instead of being confined to a disjoint
    N_symbols block where edge positions see a zero-padded fake neighbor. One continuous stream
    of val_size symbols is processed in chunks of chunk_size *newly graded* symbols each; adjacent
    chunks overlap by N-1 raw symbols (borrowed, not re-transmitted) so no window at a chunk
    boundary ever loses real context. Only the first/last N//2 symbols of the whole stream are
    excluded (they don't have a full window). Returns a single pooled BLER -- there's no
    'per-position' breakdown left to report, since every graded symbol is equally a true center.
    """
    transmitter, channel, receiver = models['transmitter'], models['channel'], models['receiver']
    device = parameters['device']
    N = parameters['N']
    overlap = N - 1

    total_windows = val_size - overlap
    if total_windows <= 0:
        raise ValueError(f"val_size ({val_size}) must be > N-1 ({overlap}) to produce at least "
                          f"one full sliding window")

    criterion = T.nn.CrossEntropyLoss()
    error_counter = ErrorCounter(parameters['bits'])
    onehot_to_int = OneHotToInt()
    generator = OneHotVecGenerator()

    labels_cpu = generator(parameters['M'], val_size)
    onehot_cpu = generator.convert_to_onehot(labels_cpu)
    labels = labels_cpu.to(device)
    onehot = onehot_cpu.to(device).type(T.float)

    channel.set_length(length=length)
    loss_meter = AverageMeter()
    errors, bits_counter, n_samples = 0, 0, 0

    with T.no_grad():
        start = 0
        with tqdm(total=total_windows, desc=f"km={length} (sliding)", leave=False) as pbar:
            while start < total_windows:
                n_new = min(chunk_size, total_windows - start)
                chunk_inputs = onehot[start:start + n_new + overlap]
                chunk_targets = labels[start + N // 2: start + N // 2 + n_new]

                t_outputs = transmitter(chunk_inputs, length)
                c_outputs = channel(t_outputs, length)
                r_outputs = receiver(c_outputs, length)

                loss = criterion(r_outputs, chunk_targets)
                loss_meter.update(loss.item(), n_new)

                preds = onehot_to_int(r_outputs.detach())
                errors_t, bits = error_counter(preds, chunk_targets)
                errors += errors_t
                bits_counter += bits
                n_samples += n_new

                start += n_new
                pbar.update(n_new)

    bler = errors.astype(np.float64) / float(bits_counter)
    details = {"n_samples": n_samples, "n_bits": bits_counter, "n_errors": int(errors),
               "n_stream_symbols_excluded_at_edges": val_size - n_samples}
    return bler, loss_meter.avg, details


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
    parameters['full_eval'] = (args.eval_mode == 'full')
    parameters['sliding_eval'] = (args.eval_mode == 'sliding')

    models = load_models(parameters, checkpoint_dir)
    print(f"eval_mode={args.eval_mode}"
          f"{' (decoding every symbol per block, not just the center)' if parameters['full_eval'] else ''}"
          f"{' (stride-1 windows over one continuous stream, true neighbors everywhere)' if parameters['sliding_eval'] else ''}")

    if not parameters['sliding_eval']:
        loader = get_random_ds(size=parameters['val_size'], max_num=parameters['M'],
                                N_symbols=parameters['N'], device=parameters['device'],
                                batch_size=parameters['batch'], dtype=T.float,
                                full_targets=parameters['full_eval'])

    rows = []
    print(f"{'km':>6}  {'BLER':>12}  {'CE Loss':>12}")
    pbar = tqdm(args.eval_kms, desc="Evaluating")
    for km in pbar:
        pbar.set_postfix(km=km)
        if parameters['sliding_eval']:
            bler, loss, details = evaluate_sliding(models, parameters, km,
                                                     val_size=parameters['val_size'],
                                                     chunk_size=parameters['batch'])
        else:
            bler, loss, details = evaluate(models, parameters, loader, km, track_positions=parameters['full_eval'])
        print(f"{km:>6}  {bler:>12.6e}  {loss:>12.6f}")
        rows.append((km, bler, loss))

        print(f"  samples evaluated: {details['n_samples']:,}   bits evaluated: {details['n_bits']:,}   "
              f"bit errors: {details['n_errors']:,}")
        if 'n_stream_symbols_excluded_at_edges' in details:
            print(f"  stream symbols excluded (no full window, at the very start/end of the "
                  f"whole stream): {details['n_stream_symbols_excluded_at_edges']}")
        if 'per_position_bler' in details:
            center = parameters['N'] // 2
            print(f"  per-position BLER (position {center} = the label 'center' mode trains/grades):")
            for p, b in enumerate(details['per_position_bler']):
                marker = "  <- center" if p == center else ""
                print(f"    pos {p:2d}: {b:.6e}{marker}")

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
