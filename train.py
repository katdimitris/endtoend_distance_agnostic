import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from trainer import Trainer
from evaluator import Evaluator
from parameters import get_channel_parameters, get_models
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--bits", type=int, default=6)
parser.add_argument("--n", type=int, default=48)
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

parser.add_argument("--optimizer", type=str, default='adam')
parser.add_argument("--scheduler", type=str, default='cosine')
parser.add_argument("--lr", type=float, default=0.0005)

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
parser.add_argument("--film_layers",type=str, default="0", help="receiver FiLM layer indices, e.g. 0,1,2")
parser.add_argument("--film_mode",type=str, default="both", help="both, gamma, beta")
parser.add_argument("--mlp_layers",type=int, default=1)

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
    parameters['batch'] = args.batch_size
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
    parameters['d'] = args.d
    parameters['sigma'] = args.sigma
    parameters['step'] = args.step
    parameters['transmitter_type'] = args.transmitter_type
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

    exp_dir = f'{args.exp_dir}_train={args.train_lengths}_eval={args.transfer_lengths}/'
    exp_name = f'{method_str}_it={args.train_iter}_vs={args.val_size}_opt={args.optimizer}_lr={args.lr}_bs={args.batch_size}_s={args.seed}'
    return exp_dir, exp_name


def main():
    parameters = build_parameters(args)
    exp_dir, exp_name = build_exp_identifiers(args, parameters)

    writer = SummaryWriter(f'hardware_exps_v2/{exp_dir}/{exp_name}')

    models = get_models(parameters)
    trainer = Trainer(models, parameters, writer, exp_dir, exp_name)
    trainer.run()
    trainer.save()


if __name__ == '__main__':
    args = parser.parse_args()
    fix_seed(args.seed)
    main()