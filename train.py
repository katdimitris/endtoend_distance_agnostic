import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from trainer import Trainer
from evaluator import Evaluator
from parameters import get_channel_parameters, get_models
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--bits", type=int, default=6)
parser.add_argument("--n", type=int, default=40)
parser.add_argument("--exp_dir", type=str, default='hardware_exp_debug_0_55')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--logging', type=str2bool, default=True)

parser.add_argument("--transfer_lengths", type=lambda s: tuple(map(int, s.split(','))), default=(0, 55))
parser.add_argument("--train_lengths", type=lambda s: tuple(map(int, s.split(','))), default=(0, 55))
parser.add_argument("--train_km_list", type=lambda s: tuple(int(x) for x in s.split(',')) if s else None,
                     default=None,
                     help="If set, each training sample's length is drawn uniformly at random (with "
                          "replacement) from this discrete km list instead of continuously from train_lengths, "
                          "e.g. '0,10,17,25,35,42'.")
parser.add_argument("--eval_step", type=int, default=1,
                     help="km spacing of the post-training evaluation sweep over transfer_lengths.")
parser.add_argument("--no_eval", type=str2bool, default=False,
                     help="skip the post-training evaluation sweep entirely (train-only, faster).")
parser.add_argument("--resume_from", type=str, default=None,
                     help="checkpoint dir (as saved by Trainer.save) to load transmitter/channel/"
                          "receiver weights from before training starts, e.g. for progressively "
                          "activating more film_layers on top of an already-converged model. "
                          "Architecture must match except (optionally) film_layers -- all film "
                          "heads exist regardless of which are active, so activating a previously-"
                          "inactive one just means it starts from its own original untrained init "
                          "while the rest of the model resumes already-converged. Optimizer/LR "
                          "schedule always start fresh over this run's --train_iter budget.")

parser.add_argument("--optimizer", type=str, default='adam')
parser.add_argument("--scheduler", type=str, default='cosine')
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--loss", type=str, default='ce', choices=['ce', 'focal'],
                     help="training loss: 'ce' (cross-entropy, default) or 'focal' (utils.FocalLoss, "
                          "down-weights confidently-decoded samples so the hard long-km ones dominate).")
parser.add_argument("--focal_gamma", type=float, default=2.0,
                     help="focusing exponent for --loss focal (0 == plain CE). Ignored for --loss ce.")
parser.add_argument("--fixed_sigma", type=float, default=None,
                     help="if set, the AWGN std is this constant at EVERY length instead of the per-km "
                          "std_km_table interpolation (parameters.py) -- noise no longer grows with km.")
parser.add_argument("--sigma_scale", type=float, default=None,
                    help="multiply every AWGN std in std_km_table by this factor (e.g. 1.5 = +50%%); with the "
                         "log-linear interpolation this scales the whole std(km) curve, extrapolation included.")
parser.add_argument("--er", type=float, default=None,
                    help="modulator extinction ratio in dB (default: parameters.py, 20). Applied to the field "
                         "min-max range, like the default.")
parser.add_argument("--snr", type=float, default=None,
                    help="DAC/ADC quantization-noise SNR in dB (default: parameters.py, 38).")

parser.add_argument("--receiver_type", type=str, default='baseline')
parser.add_argument("--transmitter_type", type=str, default='baseline')
parser.add_argument("--length_encoding", type=str, default='one_hot')
parser.add_argument("--d", type=int, default=20)
parser.add_argument("--sigma", type=float, default=2.0)
parser.add_argument("--step", type=float, default=2.0)
parser.add_argument("--position", type=str, default='preact')
parser.add_argument("--train_iter", type=int, default=20000)  # 1000
parser.add_argument("--val_iter", type=int, default=1) # 100
parser.add_argument("--train_size", type=int, default=10000000)  # 253000 506000 Papers: 250 * 1e5 25300 25000000 1012000
parser.add_argument("--val_size", type=int, default=616000)  #  154000 616000
parser.add_argument('--batch_size', type=int, default=1012) # 2024 1012 506
parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers for the training loader (0 = in-process)')
parser.add_argument("--film_layers",type=str, default="0", help="receiver FiLM layer indices, e.g. 0,1,2")
parser.add_argument("--film_mode",type=str, default="both", help="both, gamma, beta")
parser.add_argument("--mlp_layers",type=int, default=1)
parser.add_argument("--window_margin", type=int, default=2,
                     help="for receiver_type=equalizer: number of neighbor symbols on each side of "
                          "the center the receiver sees (WindowDeserializer). Ignored otherwise.")
parser.add_argument("--rank", type=int, default=8,
                     help="for receiver_type=equalizer: rank of the length-conditioned low-rank "
                          "weight correction on the receiver's first layer. Ignored otherwise.")

ALLOWED_ACTIVATIONS = ('relu', 'sigmoid', 'photonic_sigmoid', 'cliprelu')
parser.add_argument("--transmitter_activations", type=str, default="relu,relu,sigmoid",
                     help="comma-separated, exactly 3 values: activation after layer_in, layer_1, "
                          f"layer_out. Each one of {ALLOWED_ACTIVATIONS}.")
parser.add_argument("--receiver_activations", type=str, default="relu,relu",
                     help="comma-separated, exactly 2 values: activation after layer_in, layer_1 "
                          f"(receiver/FiLMReceiver has no output activation). Each one of {ALLOWED_ACTIVATIONS}.")


def parse_activations(s, n, argname):
    parts = [p.strip().lower() for p in s.split(',')]
    if len(parts) != n:
        raise ValueError(f"--{argname} must have exactly {n} comma-separated values, got {len(parts)}: {s!r}")
    bad = [p for p in parts if p not in ALLOWED_ACTIVATIONS]
    if bad:
        raise ValueError(f"--{argname} has invalid activation(s) {bad}; allowed: {ALLOWED_ACTIVATIONS}")
    return tuple(parts)


def build_parameters(args):
    """Everything needed by get_models()/Trainer, built from parsed CLI args.
    Shared by train.py and inference.py so both reconstruct the identical architecture."""
    device = T.device(f"cuda:{args.gpu_id}" if T.cuda.is_available() else "cpu")
    parameters = get_channel_parameters()

    parameters['M'] = int(2 ** args.bits)
    parameters['n'] = args.n
    parameters['bits'] = args.bits
    parameters['device'] = device
    parameters['seed'] = args.seed
    parameters['log_interval'] = args.log_interval
    parameters['logging'] = args.logging

    parameters['train_size'] = args.train_size
    parameters['train_iter'] = args.train_iter
    parameters['val_size'] = args.val_size
    parameters['val_iter'] = args.val_iter
    parameters['lr'] = args.lr
    parameters['loss'] = args.loss
    parameters['focal_gamma'] = args.focal_gamma if args.loss == 'focal' else None
    if args.fixed_sigma is not None:
        parameters['std_km_table'] = ((0, args.fixed_sigma),)  # km is irrelevant for 'constant'
        parameters['std_interp'] = 'constant'
    if args.sigma_scale is not None:
        parameters['std_km_table'] = tuple((km, std * args.sigma_scale) for km, std in parameters['std_km_table'])
    if args.er is not None:
        parameters['ER'] = args.er
    if args.snr is not None:
        parameters['snr'] = args.snr
    parameters['batch'] = args.batch_size
    parameters['num_workers'] = args.num_workers
    parameters['optimizer'] = args.optimizer
    parameters['receiver_type'] = args.receiver_type
    parameters['train_lengths'] = args.train_lengths
    parameters['transfer_lengths'] = args.transfer_lengths
    parameters['train_km_list'] = args.train_km_list
    parameters['eval_step'] = args.eval_step
    parameters['no_eval'] = args.no_eval
    parameters['length_encoding'] = args.length_encoding
    parameters['position'] = args.position
    film_layers = tuple(int(x) for x in args.film_layers.split(",") if x != '')
    parameters['film_layers'] = film_layers
    parameters['film_mode'] = args.film_mode
    parameters['mlp_layers'] = args.mlp_layers
    # window_margin drives BOTH the channel's WindowDeserializer and the equalizer receiver's
    # input width, so they must agree -- only set it for the equalizer, else None (center-only).
    parameters['window_margin'] = args.window_margin if args.receiver_type in ('equalizer', 'wide') else None
    parameters['rank'] = args.rank
    parameters['d'] = args.d
    parameters['sigma'] = args.sigma
    parameters['step'] = args.step
    parameters['transmitter_type'] = args.transmitter_type
    parameters['transmitter_activations'] = parse_activations(args.transmitter_activations, 3, 'transmitter_activations')
    parameters['receiver_activations'] = parse_activations(args.receiver_activations, 2, 'receiver_activations')
    return parameters


def build_exp_identifiers(args, parameters):
    """(exp_dir, exp_name) matching train.py's naming convention, from the same args/parameters
    build_parameters() was given. Shared by train.py and inference.py so a checkpoint saved by
    one is found by the other without duplicating (and risking drifting from) this naming logic."""
    film_layers = parameters['film_layers']
    if 'tags' in parameters['receiver_type']:
        method_str = 'tags'
        if parameters['length_encoding'] == 'gaussian':
            method_str += f'_{args.length_encoding}_d={args.d}_s={args.sigma}'
        elif parameters['length_encoding'] == 'one_hot':
            method_str += f'_{args.length_encoding}_step={args.step}'
    elif 'film' in parameters['receiver_type']:
        method_str = 'film'
        method_str += f'_{args.length_encoding}_mode={args.film_mode}_mlp={args.mlp_layers}_pos={args.position}_l={film_layers}'
    elif parameters['receiver_type'] == 'equalizer':
        method_str = f'equalizer_{args.length_encoding}_margin={args.window_margin}_rank={args.rank}'
    elif parameters['receiver_type'] == 'wide':
        method_str = f'wide_margin={args.window_margin}'
    else:
        method_str = 'baseline'

    if args.transmitter_type == 'film':
        if 'film' in parameters['receiver_type']:
            # receiver's FiLM descriptor above already encodes mode/mlp/pos/layers,
            # assumed shared with the transmitter's FiLM config in this run.
            method_str += '_transm_film'
        else:
            # Tx-only FiLM: receiver side never printed a FiLM descriptor, so print
            # one here -- otherwise every Tx-only config collapses to 'baseline_transm_film'
            # and different mode/mlp/layers runs silently overwrite each other's logs.
            method_str += f'_transm_film_{args.length_encoding}_mode={args.film_mode}_mlp={args.mlp_layers}_pos={args.position}_l={film_layers}'
    if args.train_km_list:
        method_str += '_trainkms=' + '-'.join(str(k) for k in args.train_km_list)
    if args.loss != 'ce':
        # CE runs keep their historical names; any other loss must never overwrite them
        method_str += f'_loss={args.loss}_g={args.focal_gamma}'
    if args.fixed_sigma is not None:
        method_str += f'_sigma={args.fixed_sigma}'
    if args.sigma_scale is not None:
        method_str += f'_sigmascale={args.sigma_scale}'
    if args.er is not None:
        method_str += f'_ER={args.er}'
    if args.snr is not None:
        method_str += f'_snr={args.snr}'
    if args.resume_from:
        # so a resumed run never collides with a from-scratch run of the same film_layers
        method_str += '_resumed'

    exp_dir = f'{args.exp_dir}_train={args.train_lengths}_eval={args.transfer_lengths}/'
    exp_name = f'{method_str}_it={args.train_iter}_vs={args.val_size}_opt={args.optimizer}_lr={args.lr}_bs={args.batch_size}_s={args.seed}'
    return exp_dir, exp_name


def main():
    parameters = build_parameters(args)
    parameters['resumed_from'] = args.resume_from
    exp_dir, exp_name = build_exp_identifiers(args, parameters)

    writer = SummaryWriter(f'hardware_exps_v2/{exp_dir}/{exp_name}')

    models = get_models(parameters)
    if args.resume_from:
        print(f"Resuming weights from {args.resume_from}")
        for name, model in models.items():
            state = T.load(f'{args.resume_from}/{name}.pt', map_location=parameters['device'], weights_only=True)
            model.load_state_dict(state)

    trainer = Trainer(models, parameters, writer, exp_dir, exp_name)
    trainer.run()
    trainer.save()


if __name__ == '__main__':
    args = parser.parse_args()
    fix_seed(args.seed)
    main()