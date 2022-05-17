# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import json
import itertools
import logging
import os
import numpy as np

from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    encoders,
    indexed_dataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
)

from fairseq.tasks import FairseqTask, register_task
from fairseq import search
from data.data2text_dataset import Data2TextDataset

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


def load_data2text_dataset(
    data_path, split,
    field, field_dict,
    lpos, lpos_dict,
    rpos, rpos_dict,
    src, src_dict,
    kwd, tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False,
    truncate_source=False
):

    def split_exists(split, middle_tuple, lang, data_path):
        middle = '-'.join(middle_tuple)
        filename = os.path.join(data_path, '{}.{}.{}'.format(split, middle, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    field_datasets = []
    lpos_datasets = []
    rpos_datasets = []
    src_datasets = []
    kwd_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if kwd is not None and split_exists(split_k, (field, lpos, rpos, src, kwd, tgt), src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}-{}-{}-{}-{}.'.format(split_k, field, lpos, rpos, src, kwd, tgt))
        elif split_exists(split_k, (field, lpos, rpos, src, tgt), src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}-{}-{}-{}.'.format(split_k, field, lpos, rpos, src, tgt))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        field_dataset = data_utils.load_indexed_dataset(prefix + field, field_dict, dataset_impl)
        lpos_dataset = data_utils.load_indexed_dataset(prefix + lpos, lpos_dict, dataset_impl)
        rpos_dataset = data_utils.load_indexed_dataset(prefix + rpos, rpos_dict, dataset_impl)
        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        if truncate_source:
            field_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(field_dataset, field_dict.eos()),
                    max_source_positions - 1,
                ),
                field_dict.eos(),
            )
            lpos_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(lpos_dataset, lpos_dict.eos()),
                    max_source_positions - 1,
                ),
                lpos_dict.eos(),
            )
            rpos_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(rpos_dataset, rpos_dict.eos()),
                    max_source_positions - 1,
                ),
                rpos_dict.eos(),
            )
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        field_datasets.append(field_dataset)
        lpos_datasets.append(lpos_dataset)
        rpos_datasets.append(rpos_dataset)
        src_datasets.append(src_dataset)

        if kwd is not None:
            kwd_dataset = data_utils.load_indexed_dataset(prefix + kwd, tgt_dict, dataset_impl)
            kwd_datasets.append(kwd_dataset)
        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        if kwd is not None:
            logger.info('{} {} {}-{}-{}-{}-{}-{} {} examples'.format(
                data_path, split_k, field, lpos, rpos, src, kwd, tgt, len(src_datasets[-1])
            ))
        else:
            logger.info('{} {} {}-{}-{}-{}-{} {} examples'.format(
                data_path, split_k, field, lpos, rpos, src, tgt, len(src_datasets[-1])
            ))

        if not combine:
            break

    assert len(field_datasets) == len(lpos_datasets) == len(rpos_datasets) == \
           len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        field_dataset = field_datasets[0]
        lpos_dataset = lpos_datasets[0]
        rpos_dataset = rpos_datasets[0]
        src_dataset = src_datasets[0]
        kwd_dataset = kwd_datasets[0] if len(kwd_datasets) > 0 else None
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        field_dataset = ConcatDataset(field_datasets, sample_ratios)
        lpos_dataset = ConcatDataset(lpos_datasets, sample_ratios)
        rpos_dataset = ConcatDataset(rpos_datasets, sample_ratios)
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        kwd_dataset = ConcatDataset(kwd_datasets, sample_ratios) if len(kwd_datasets) > 0 else None
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios) if len(tgt_datasets) > 0 else None

    if prepend_bos:
        assert hasattr(field_dict, "bos_index") and \
               hasattr(lpos_dict, "bos_index") and hasattr(rpos_dict, "bos_index") and \
               hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        field_dataset = PrependTokenDataset(field_dataset, field_dict.bos())
        lpos_dataset = PrependTokenDataset(lpos_dataset, lpos_dict.bos())
        rpos_dataset = PrependTokenDataset(rpos_dataset, rpos_dict.bos())
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if kwd_dataset is not None:
            kwd_dataset = PrependTokenDataset(kwd_dataset, tgt_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    kwd_dataset_sizes = kwd_dataset.sizes if kwd_dataset is not None else None
    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return Data2TextDataset(
        field_dataset, field_dict,
        lpos_dataset, lpos_dict,
        rpos_dataset, rpos_dict,
        src_dataset, src_dataset.sizes, src_dict,
        kwd_dataset, kwd_dataset_sizes,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions
    )


@register_task('data2text')
class Data2TextTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-f', '--field-lang', default=None, metavar='Field',
                            help='field input')
        parser.add_argument('-l', '--lpos-lang', default=None, metavar='LPOS',
                            help='lpos input')
        parser.add_argument('-r', '--rpos-lang', default=None, metavar='RPOS',
                            help='rpos input')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-k', '--kwd-lang', default=None, metavar='KEYWORD',
                            help='prefix keyword')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenizer before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='if setting, we compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        # fmt: on

    def __init__(self, args, field_dict, lpos_dict, rpos_dict, src_dict, tgt_dict):
        super().__init__(args)
        self.field_dict = field_dict
        self.lpos_dict = lpos_dict
        self.rpos_dict = rpos_dict
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.sep = self.tgt_dict.index('<sep>')
        self.ins = self.tgt_dict.index('<ins>')
        assert self.src_dict.index('<sep>') == self.tgt_dict.index('<sep>')
        assert self.src_dict.index('<ins>') == self.tgt_dict.index('<ins>')

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0

        # load dictionaries
        field_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.field_lang)))
        lpos_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.lpos_lang)))
        rpos_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.rpos_lang)))
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert field_dict.bos() == lpos_dict.bos() == rpos_dict.bos() == src_dict.bos() == tgt_dict.bos()
        assert field_dict.pad() == lpos_dict.pad() == rpos_dict.pad() == src_dict.pad() == tgt_dict.pad()
        assert field_dict.eos() == lpos_dict.eos() == rpos_dict.eos() == src_dict.eos() == tgt_dict.eos()
        assert field_dict.unk() == lpos_dict.unk() == rpos_dict.unk() == src_dict.unk() == tgt_dict.unk()
        assert src_dict.index('<sep>') == tgt_dict.index('<sep>')
        assert src_dict.index('<ins>') == tgt_dict.index('<ins>')
        logger.info('[{}] dictionary: {} types'.format(args.field_lang, len(field_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.lpos_lang, len(lpos_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.rpos_lang, len(rpos_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, field_dict, lpos_dict, rpos_dict, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        field, lpos, rpos, src, kwd, tgt = self.args.field_lang, self.args.lpos_lang, self.args.rpos_lang, \
                                           self.args.source_lang, self.args.kwd_lang, self.args.target_lang

        self.datasets[split] = load_data2text_dataset(
            data_path, split, field, self.field_dict,
            lpos, self.lpos_dict, rpos, self.rpos_dict,
            src, self.src_dict, kwd, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            truncate_source=self.args.truncate_source,
        )

    def build_dataset_for_inference(
            self, field_tokens, lpos_tokens, rpos_tokens, src_tokens, src_lengths, kwd_tokens=None, kwd_lengths=None
    ):
        return Data2TextDataset(
            field_tokens, self.field_dictionay,
            lpos_tokens, self.lpos_dictionary,
            rpos_tokens, self.rpos_dictionary,
            src_tokens, src_lengths, self.source_dictionary,
            kwd_tokens, kwd_lengths,
            append_bos=True if kwd_tokens is not None else False
        )

    def build_generator(self, models, args):

        from models.sketch_model.sequence_generator import (
            SequenceGenerator,
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
        )

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator([model], Namespace(**gen_args))
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                count = sum_logs('_bleu_counts_' + str(i))
                count = count if isinstance(count, int) else count.cpu()
                total = sum_logs('_bleu_totals_' + str(i))
                total = total if isinstance(total, int) else total.cpu()
                counts.append(count)
                totals.append(total)

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar('_bleu_counts', np.array(counts))
                metrics.log_scalar('_bleu_totals', np.array(totals))
                metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
                metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu
                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if 'smooth_method' in fn_sig:
                        smooth = {'smooth_method': 'exp'}
                    else:
                        smooth = {'smooth': 'exp'}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters['_bleu_counts'].sum,
                        total=meters['_bleu_totals'].sum,
                        sys_len=meters['_bleu_sys_len'].sum,
                        ref_len=meters['_bleu_ref_len'].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived('bleu', compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def field_dictionary(self):
        return self.field_dict

    @property
    def lpos_dictionary(self):
        return self.lpos_dict

    @property
    def rpos_dictionary(self):
        return self.rpos_dict

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
                ),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        kwd_tokens = sample['kwd_tokens'] if 'kwd_tokens' in sample else None
        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=kwd_tokens)
        kwds, hyps, refs = [], [], []
        for i in range(len(gen_out)):
            if kwd_tokens is not None:
                kwds.append(decode(
                    utils.strip_pad(gen_out[i][0]['kwd_tokens'], self.tgt_dict.pad()),
                    escape_unk=True,
                ))
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
        if self.args.eval_bleu_print_samples:
            if kwd_tokens is not None:
                logger.info('example keywords: ' + kwds[0])
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
