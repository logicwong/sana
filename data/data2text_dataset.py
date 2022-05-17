# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch

from fairseq.data import data_utils, FairseqDataset


logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge(
        'source', left_pad=left_pad_source,
        pad_to_length=pad_to_length['source'] if pad_to_length is not None else None
    )
    # sort by descending source length
    src_lengths = torch.LongTensor([
        s['source'].ne(pad_idx).long().sum() for s in samples
    ])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    field_tokens = merge('field', left_pad=left_pad_source)
    field_tokens = field_tokens.index_select(0, sort_order)

    lpos_tokens = merge('lpos', left_pad=left_pad_source)
    lpos_tokens = lpos_tokens.index_select(0, sort_order)

    rpos_tokens = merge('rpos', left_pad=left_pad_source)
    rpos_tokens = rpos_tokens.index_select(0, sort_order)

    kwd_tokens = None
    if samples[0].get('kwd', None) is not None:
        kwd_tokens = merge(
            'kwd', left_pad=left_pad_target,
            pad_to_length=pad_to_length['kwd'] if pad_to_length is not None else None,
        )
        kwd_tokens = kwd_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge(
            'target', left_pad=left_pad_target,
            pad_to_length=pad_to_length['target'] if pad_to_length is not None else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([
            s['target'].ne(pad_idx).long().sum() for s in samples
        ]).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get('prev_output_tokens', None) is not None:
            prev_output_tokens = merge('prev_output_tokens', left_pad=left_pad_target)
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length['target'] if pad_to_length is not None else None,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'field_tokens': field_tokens,
            'lpos_tokens': lpos_tokens,
            'rpos_tokens': rpos_tokens,
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if kwd_tokens is not None:
        batch['kwd_tokens'] = kwd_tokens
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class Data2TextDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
    """

    def __init__(
        self,
        field, field_dict,
        lpos, lpos_dict,
        rpos, rpos_dict,
        src, src_sizes, src_dict,
        kwd=None, kwd_sizes=None,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        append_bos=False, eos=None,
        src_lang_id=None,
        tgt_lang_id=None,
    ):
        if tgt_dict is not None:
            assert field_dict.pad() == lpos_dict.pad() == rpos_dict.pad() == src_dict.pad() == tgt_dict.pad()
            assert field_dict.eos() == lpos_dict.eos() == rpos_dict.eos() == src_dict.eos() == tgt_dict.eos()
            assert field_dict.unk() == lpos_dict.unk() == rpos_dict.unk() == src_dict.unk() == tgt_dict.unk()
        self.field = field
        self.lpos = lpos
        self.rpos = rpos
        self.src = src
        self.kwd = kwd
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.kwd_sizes = np.array(kwd_sizes) if kwd_sizes is not None else None
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.field_dict = field_dict
        self.lpos_dict = lpos_dict
        self.rpos_dict = rpos_dict
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.append_bos = append_bos
        self.eos = (eos if eos is not None else src_dict.eos())
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        kwd_item = self.kwd[index] if self.kwd is not None else None
        src_item = self.src[index]
        field_item = self.field[index]
        lpos_item = self.lpos[index]
        rpos_item = self.rpos[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.kwd and self.kwd[index][-1] != eos:
                kwd_item = torch.cat([self.kwd[index], torch.LongTensor([eos])])
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.kwd and self.kwd[index][0] != bos:
                kwd_item = torch.cat([torch.LongTensor([bos]), self.kwd[index]])
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.field[index][0] != bos:
                field_item = torch.cat([torch.LongTensor([bos]), self.field[index]])
            if self.lpos[index][0] != bos:
                lpos_item = torch.cat([torch.LongTensor([bos]), self.lpos[index]])
            if self.rpos[index][0] != bos:
                rpos_item = torch.cat([torch.LongTensor([bos]), self.rpos[index]])
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.field[index][-1] == eos:
                field_item = self.field[index][:-1]
            if self.lpos[index][-1] == eos:
                lpos_item = self.lpos[index][:-1]
            if self.rpos[index][-1] == eos:
                rpos_item = self.rpos[index][:-1]
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        assert len(field_item) == len(lpos_item) == len(rpos_item) == len(src_item)

        example = {
            'id': index,
            'field': field_item,
            'lpos': lpos_item,
            'rpos': rpos_item,
            'source': src_item,
            'kwd': kwd_item,
            'target': tgt_item,
        }
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res['net_input']['src_tokens']
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res['net_input']['src_lang_id'] = torch.LongTensor(
                            [[self.src_lang_id]]
                            ).expand(bsz, 1).to(src_tokens)
            if self.tgt_lang_id is not None:
                res['tgt_lang_id'] = torch.LongTensor(
                            [[self.tgt_lang_id]]
                            ).expand(bsz, 1).to(src_tokens)
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index],
                   self.kwd_sizes[index] if self.kwd_sizes is not None else 0,
                   self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index],
                self.kwd_sizes[index] if self.kwd_sizes is not None else 0,
                self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
                getattr(self.field, 'supports_prefetch', False)
                and getattr(self.lpos, 'supports_prefetch', False)
                and getattr(self.rpos, 'supports_prefetch', False)
                and getattr(self.src, 'supports_prefetch', False)
                and (getattr(self.kwd, 'supports_prefetch', False) or self.kwd is None)
                and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.field.prefetch(indices)
        self.lpos.prefetch(indices)
        self.rpos.prefetch(indices)
        self.src.prefetch(indices)
        if self.kwd is not None:
            self.kwd.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """ Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        if max_sizes is None:
            return indices, []
        if type(max_sizes) in (int, float):
            max_src_size, max_tgt_size = max_sizes, max_sizes
        else:
            max_src_size, max_tgt_size = max_sizes
        if self.tgt_sizes is None:
            ignored = indices[self.src_sizes[indices] > max_src_size]
        else:
            ignored = indices[(self.src_sizes[indices] > max_src_size) |
                              (self.tgt_sizes[indices] > max_tgt_size)]
        if len(ignored) > 0:
            if self.tgt_sizes is None:
                indices = indices[self.src_sizes[indices] <= max_src_size]
            else:
                indices = indices[(self.src_sizes[indices] <= max_src_size) &
                                  (self.tgt_sizes[indices] <= max_tgt_size)]
        return indices, ignored.tolist()
