# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq.utils import new_arange
from fairseq.tasks import register_task
from fairseq import utils

from models.refinement_model.levenshtein_utils import _get_kwd_del_masks, _get_del_targets, _apply_del_words
from tasks.data2text import Data2TextTask, load_data2text_dataset, Data2TextDataset


@register_task('refinement')
class RefinementTask(Data2TextTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        Data2TextTask.add_args(parser)
        parser.add_argument(
            '--noise',
            default='random_delete',
            choices=['random_delete', 'random_mask', 'no_noise', 'full_mask'])
        parser.add_argument('--beta', type=float, default=1.0, help='roll-in-policy beta')
        parser.add_argument('--behind-penalty', type=float, default=0.0, help='penalty target words behind last keyword')

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        field, lpos, rpos, src, kwd, tgt = self.args.field_lang, self.args.lpos_lang, self.args.rpos_lang, \
                                           self.args.source_lang, self.args.kwd_lang, self.args.target_lang

        self.datasets[split] = load_data2text_dataset(
            data_path, split, field, self.field_dict, lpos, self.lpos_dict, rpos, self.rpos_dict,
            src, self.src_dict, kwd, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            prepend_bos=True,
        )

    def inject_noise(self, target_tokens, kwd_tokens):
        def _random_delete(target_tokens, prev_mask=None):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            dot = self.tgt_dict.index('.')

            roll_in_mask = target_tokens.clone().float().uniform_() < self.args.beta
            prev_mask = prev_mask.masked_fill(~roll_in_mask, False)
            target_mask = target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
            _mask = target_mask.new_zeros(target_mask.size()).bool()
            _mask[:, -2] = True
            target_mask = target_mask | (_mask & target_mask.eq(dot))

            target_score = target_tokens.clone().float().uniform_()
            if self.args.behind_penalty > 0.0:
                target_orders = new_arange(target_tokens)
                prev_orders = target_tokens.new_zeros(*target_tokens.size()).masked_scatter(prev_mask, target_orders)
                post_mask = target_orders > prev_orders.argmax(1, keepdim=True)
                target_score = target_score.masked_scatter(post_mask, target_score - self.args.behind_penalty)
            target_score.masked_fill_(prev_mask, 1.5)
            target_score.masked_fill_(target_mask, 2.0)

            delete_length = (~(prev_mask | target_mask)).sum(1).float()
            delete_length = delete_length * delete_length.clone().uniform_()
            delete_length = delete_length + 1  # make sure to delete at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < delete_length[:, None].long()

            max_len = target_tokens.size(1)
            prev_target_tokens = target_tokens.gather(
                1, target_rank).masked_fill_(target_cutoff, pad).gather(
                    1,
                    target_rank.masked_fill_(target_cutoff,
                                             max_len).sort(1)[1])
            prev_target_tokens = prev_target_tokens[:, :prev_target_tokens.
                                                    ne(pad).sum(1).max()].contiguous()

            return prev_target_tokens

        def _random_mask(target_tokens, prev_mask=None):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            ins = self.ins

            roll_in_mask = target_tokens.clone().float().uniform_() < self.args.beta
            target_mask = target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
            target_mask = target_mask | prev_mask.masked_fill(~roll_in_mask, False)

            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(target_mask, 2.0)
            mask_length = (~target_mask).sum(1).float()
            mask_length = mask_length * mask_length.clone().uniform_()
            mask_length = mask_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < mask_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), ins)

            return prev_target_tokens

        def _full_mask(target_tokens, prev_mask=None):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            ins = self.ins

            roll_in_mask = target_tokens.clone().float().uniform_() < self.args.beta
            target_mask = target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
            target_mask = target_mask | prev_mask.masked_fill(~roll_in_mask, False)

            return target_tokens.masked_fill(~target_mask, ins)

        prev_mask = _get_kwd_del_masks(
            target_tokens, kwd_tokens, self.tgt_dict.bos(), self.tgt_dict.eos(),
            self.tgt_dict.pad(), self.sep
        )

        if self.args.noise == 'random_delete':
            return _random_delete(target_tokens, prev_mask)
        elif self.args.noise == 'random_mask':
            return _random_mask(target_tokens, prev_mask)
        elif self.args.noise == 'full_mask':
            return _full_mask(target_tokens, prev_mask)
        elif self.args.noise == 'no_noise':
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, models, args):
        from models.refinement_model.iterative_refinement_generator import IterativeRefinementGenerator
        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
            max_iter=getattr(args, 'iter_decode_max_iter', 15),
            no_delete_kwd=getattr(args, 'no_delete_kwd', False),
            no_insert_kwd=getattr(args, 'no_insert_kwd', False),
            beam_size=getattr(args, 'iter_decode_with_beam', 1),
            reranking=getattr(args, 'iter_decode_with_external_reranker', False),
            decoding_format=getattr(args, 'decoding_format', None),
            adaptive=not getattr(args, 'iter_decode_force_max_iter', False),
            retain_history=getattr(args, 'retain_iter_history', False))

    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   update_num,
                   ignore_grad=False):
        model.train()
        kwd_del_targets = _get_del_targets(sample['kwd_tokens'], sample['target'], self.tgt_dict.pad())
        prefix_tokens, _, _ = _apply_del_words(
            sample['kwd_tokens'], None, None, kwd_del_targets.type(torch.bool),
            self.tgt_dict.pad(), self.tgt_dict.bos(), self.tgt_dict.eos(),
        )
        sample['prev_target'] = self.inject_noise(sample['target'], prefix_tokens)
        loss, sample_size, logging_output = criterion(model, sample, update_num)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        kwd_del_targets = _get_del_targets(sample['kwd_tokens'], sample['target'], self.tgt_dict.pad())
        prefix_tokens, _, _ = _apply_del_words(
            sample['kwd_tokens'], None, None, kwd_del_targets.type(torch.bool),
            self.tgt_dict.pad(), self.tgt_dict.bos(), self.tgt_dict.eos(),
        )
        sample['prev_target'] = self.inject_noise(sample['target'], prefix_tokens)

        return super().valid_step(sample, model, criterion)
