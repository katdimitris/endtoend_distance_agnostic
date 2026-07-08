import copy

import torch as T
import torch.nn as nn
import numpy as np

from utils import get_random_ds
from utils import ErrorCounter, OneHotToInt
from utils import *

class Evaluator():
    def __init__(self, trainer, parameters, writer, exp_dir, exp_name):
        self.exp_dir = exp_dir
        self.exp_name = exp_name
        self.writer = writer
        self.prefix = ''
        self.parameters = parameters

        self.models = trainer.models
        self.transmitter = trainer.models['transmitter']
        self.channel = trainer.models['channel']
        self.receiver = trainer.models['receiver']

        if self.parameters['n_receivers'] > 1:
            self.receiver = trainer.receiver
            self.receiver_idx = trainer.receiver_idx


        if self.parameters['tags_encoding'] == 'deep_gaussian':
            self.deep_gaussian = trainer.deep_gaussian

        if self.parameters['attention_tags']:
            self.transmitter_attention_module = trainer.transmitter_attention_module
            self.receiver_attention_module = trainer.receiver_attention_module
            self.transmitter_ref_table = trainer.transmitter_ref_table
            self.receiver_ref_table = trainer.receiver_ref_table

        self.device = parameters['device']
        self.loader = get_random_ds(size=parameters['val_size'],
                                             max_num=parameters['M'],
                                             N_symbols=parameters['N'],
                                             device=self.device,
                                             batch_size=parameters['batch'],
                                             dtype=T.float)

        self.criterion = nn.CrossEntropyLoss()
        self.error_counter = ErrorCounter(6)
        self.onehot_to_int = OneHotToInt()

    def eval(self, epoch=None, length=None):

        self.transmitter.eval()
        self.channel.eval()
        self.receiver.eval()

        update_length = True
        if length is not None:
            self.channel.set_length(length)
            update_length = False

        loss_meter = AverageMeter()
        errors, bits_counter = 0, 0
        for (inputs, targets) in self.loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            length = self.channel.get_length(update=update_length)
            if self.parameters['receiver_tags'] or self.parameters['transmitter_tags']:
                if self.parameters['tags_encoding'] == 'deep_gaussian':
                    tag = self.deep_gaussian(length)
                else:
                    tag = get_closest_tag(length, self.parameters)
                    tag = tag.to(inputs.dtype).to(self.device)

                    transmitter_tag = copy.deepcopy(tag)
                    receiver_tag = copy.deepcopy(tag)

            # transmitter tags
            if self.parameters['transmitter_tags']:
                if self.parameters['attention_tags']:
                    if self.parameters['include_repr']:
                        B, d = inputs.shape
                        n, m = self.transmitter_ref_table.shape[0], self.transmitter_ref_table.shape[1]
                        # Expand tag and ref_table
                        transmitter_tag = transmitter_tag.expand(B, -1)  # (B, d)
                        ref_table_exp = self.transmitter_ref_table.expand(B, n, m)  # (B, n, d)
                        # Create concat_tag and concat_table
                        transmitter_concat_tag = T.cat([inputs, transmitter_tag], dim=-1)  # (B, 2d)
                        transmitter_concat_table = T.cat(
                            [inputs.unsqueeze(1).expand(-1, n, -1), ref_table_exp],
                            dim=-1)  # (B, n, 2d)
                        inputs_l = self.transmitter_attention_module(transmitter_concat_tag,
                                                                     transmitter_concat_table)
                    else:
                        transmitter_tag = self.transmitter_attention_module(transmitter_tag,
                                                                            self.transmitter_ref_table)
                        inputs_l = T.cat((inputs, transmitter_tag.unsqueeze(0).expand(inputs.size(0), -1)),
                                         dim=1)
                else:
                    inputs_l = T.cat((inputs, transmitter_tag.unsqueeze(0).expand(inputs.size(0), -1)), dim=1)
            else:
                inputs_l = inputs

            t_outputs = self.transmitter(inputs_l)
            c_outputs = self.channel(t_outputs, length)

            # receiver tags
            if self.parameters['receiver_tags']:
                if self.parameters['attention_tags']:
                    if self.parameters['include_repr']:
                        B, d = c_outputs.shape
                        n, m = self.receiver_ref_table.shape[0], self.receiver_ref_table.shape[1]
                        # Expand tag and ref_table
                        receiver_tag = receiver_tag.expand(B, -1)  # (B, d)
                        ref_table_exp = self.receiver_ref_table.expand(B, n, m)  # (B, n, d)
                        # Create concat_tag and concat_table
                        receiver_concat_tag = T.cat([c_outputs, receiver_tag], dim=-1)  # (B, 2d)
                        receiver_concat_table = T.cat([c_outputs.unsqueeze(1).expand(-1, n, -1), ref_table_exp],
                                                          dim=-1)  # (B, n, 2d)
                        c_outputs_l = self.receiver_attention_module(receiver_concat_tag, receiver_concat_table)
                    else:
                        receiver_tag = self.receiver_attention_module(receiver_tag, self.receiver_ref_table)
                        c_outputs_l = T.cat((c_outputs, receiver_tag.unsqueeze(0).expand(c_outputs.size(0), -1)), dim=1)
                else:
                    c_outputs_l = T.cat((c_outputs, receiver_tag.unsqueeze(0).expand(c_outputs.size(0), -1)), dim=1)
            else:
                c_outputs_l = c_outputs

            if self.parameters['n_receivers'] == 1:
                r_outputs = self.receiver(c_outputs_l, length) # length for attn only
            else:
                r_outputs = self.receiver[self.receiver_idx(length)](c_outputs_l)

            loss = self.criterion(r_outputs, targets)
            loss_meter.update(loss.item(), inputs.size(0))

            errors_t, bits = self.error_counter(self.onehot_to_int(r_outputs.detach()), targets)
            errors += errors_t
            bits_counter += bits

        bler = errors.astype(np.float64) / float(bits_counter)
        if epoch is not None:
            if self.prefix == '':
                length_str = f' @{length}km' if not update_length else ''
                self.writer.add_scalar(f'Val{length_str}/CE Loss', loss_meter.avg, epoch)
                self.writer.add_scalar(f'Val{length_str}/BLER', bler, epoch)
            else:
                self.writer.add_scalar(f'{self.prefix}/Val CE Loss', loss_meter.avg, epoch)
                self.writer.add_scalar(f'{self.prefix}/Val BLER', bler, epoch)
        return bler, loss_meter.avg

    def get_representations(self, lengths):
        self.transmitter.eval()
        self.channel.eval()
        self.receiver.eval()

        d = {}
        for length in lengths:
            for (inputs, targets) in self.loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if length not in d:
                    d[length] = {target: {'transmitter': [], 'channel': [], 'receiver': []} for target in range(64)}

                # Get indices of the actual inputs (5th, 16th, etc.)
                actual_indices = T.arange(5, inputs.size(0), 11)  # Start at 5th (index 4), step by 11


                t_outputs = self.transmitter(inputs)
                actual_t_outputs = t_outputs[actual_indices]

                c_outputs = self.channel(t_outputs, length)
                r_outputs = self.receiver.forward_features(t_outputs)
                # Store representations in the dictionary
                for i, target in enumerate(targets):
                    target = int(target.item())  # Ensure target is an int
                    if len(d[length][target]['transmitter']) < 10:
                        d[length][target]['transmitter'].append(actual_t_outputs[i].detach().cpu().numpy())
                    if len(d[length][target]['channel']) < 10:
                        d[length][target]['channel'].append(c_outputs[i].detach().cpu().numpy())
                    if len(d[length][target]['receiver']) < 10:
                        d[length][target]['receiver'].append(r_outputs[i].detach().cpu().numpy())

        return d

