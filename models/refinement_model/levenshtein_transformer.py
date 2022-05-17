# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoderLayer,
)

from models.refinement_model.fairseq_nat_model import (
    FairseqNATModel,
    FairseqNATDecoder,
    ensemble_decoder
)
from .iterative_refinement_generator import DecoderOut
from models.data2text_transformer import base_architecture

from fairseq.modules.transformer_sentence_encoder import init_bert_params


from models.refinement_model.levenshtein_utils import (
    _skip, _skip_encoder_out, _fill,
    _get_ins_targets, _get_del_targets,
    _apply_ins_masks, _apply_ins_words, _apply_del_words,
    _get_kwd_del_masks, _get_kwd_ins_masks
)
from fairseq import utils

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

logger = logging.getLogger(__name__)

@register_model("refinement_model")
class LevenshteinTransformerModel(FairseqNATModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.num_types = len(decoder.dictionary)
        self.num_oov_types = args.source_position_markers
        self.num_embeddings = self.num_types - self.num_oov_types
        self.unk_idx = decoder.dictionary.unk()

    @property
    def allow_length_beam(self):
        return False

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)
        parser.add_argument(
            "--early-exit",
            default="6,6,6",
            type=str,
            help="number of decoder layers before word_del, mask_ins, word_ins",
        )
        parser.add_argument(
            "--no-share-discriminator",
            action="store_true",
            help="separate parameters for discriminator",
        )
        parser.add_argument(
            "--no-share-maskpredictor",
            action="store_true",
            help="separate parameters for mask-predictor",
        )
        parser.add_argument(
            "--share-discriminator-maskpredictor",
            action="store_true",
            help="share the parameters for both mask-predictor and discriminator",
        )
        parser.add_argument(
            "--sampling-for-deletion",
            action='store_true',
            help='instead of argmax, use sampling to predict the tokens'
        )


        parser.add_argument('--source-position-markers', type=int, metavar='N',
                            help='dictionary includes N additional items that '
                                 'represent an OOV token at a particular input '
                                 'position')
        parser.add_argument('--force-generation', type=float, metavar='P',
                            default=None,
                            help='set the vocabulary distribution weight to P, '
                                 'instead of predicting it from the input (1.0 '
                                 'corresponding to generation, 0.0 to pointing)')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        levenshtein_base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, 'max_source_positions', None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, 'max_target_positions', None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
        if getattr(args, 'source_position_markers', None) is None:
            args.source_position_markers = args.max_source_positions

        field_dict, lpos_dict, rpos_dict, src_dict, tgt_dict = task.field_dictionary, task.lpos_dictionary, task.rpos_dictionary, \
                                                               task.source_dictionary, task.target_dictionary
        if src_dict != tgt_dict:
            raise ValueError('Pointer-generator requires a joined dictionary')

        def build_embedding(dictionary, embed_dim, path=None, source_position_markers=0):
            # The dictionary may include additional items that can be used in
            # place of the normal OOV token and that all map to the same
            # embedding. Using a different token for each input position allows
            # one to restore the word identities from the original source text.
            num_embeddings = len(dictionary) - source_position_markers
            padding_idx = dictionary.pad()
            unk_idx = dictionary.unk()
            logger.info('dictionary indices from {0} to {1} will be mapped to {2}'
                        .format(num_embeddings, len(dictionary) - 1, unk_idx))
            emb = Embedding(num_embeddings, embed_dim, padding_idx, unk_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        field_embed_tokens = build_embedding(field_dict, args.field_embed_dim, args.field_embed_path)
        lpos_embed_tokens = build_embedding(lpos_dict, args.lpos_embed_dim, args.lpos_embed_path)
        rpos_embed_tokens = build_embedding(rpos_dict, args.rpos_embed_dim, args.rpos_embed_path)
        if args.share_all_embeddings:
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError("--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim")
            if args.decoder_embed_path and (args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError("--share-all-embeddings not compatible with --decoder-embed-path")
            encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim, args.encoder_embed_path,
                                                   args.source_position_markers)
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(src_dict, 420, args.encoder_embed_path,
                                                       args.source_position_markers)
            decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim, args.decoder_embed_path,
                                                       args.source_position_markers)

        encoder = cls.build_encoder(args, field_dict, field_embed_tokens,
                                    lpos_dict, lpos_embed_tokens, rpos_dict, rpos_embed_tokens,
                                    src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = LevenshteinTransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(
        self, field_tokens, lpos_tokens, rpos_tokens, src_tokens, src_lengths, prev_output_tokens, tgt_tokens,
            update_num, **kwargs
    ):

        assert tgt_tokens is not None, "forward function only supports training."

        # encoding
        encoder_out = self.encoder(field_tokens, lpos_tokens, rpos_tokens, src_tokens, src_lengths=src_lengths, **kwargs)

        # generate training labels for insertion
        masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_ins_targets(
            prev_output_tokens, tgt_tokens, self.pad, self.ins
        )
        mask_ins_targets = mask_ins_targets.clamp(min=0, max=255)  # for safe prediction
        mask_ins_masks = prev_output_tokens[:, 1:].ne(self.pad)

        mask_ins_out, _ = self.decoder.forward_mask_ins(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out
        )
        word_ins_out, _ = self.decoder.forward_word_ins(
            normalize=False,
            prev_output_tokens=masked_tgt_tokens,
            encoder_out=encoder_out
        )

        # make online prediction
        if self.decoder.sampling_for_deletion:
            word_predictions = torch.multinomial(
                F.softmax(word_ins_out, -1).view(-1, word_ins_out.size(-1)), 1).view(
                    word_ins_out.size(0), -1)
        else:
            word_predictions = F.log_softmax(word_ins_out, dim=-1).max(2)[1]

        word_predictions.masked_scatter_(
            ~masked_tgt_masks, tgt_tokens[~masked_tgt_masks]
        )

        # generate training labels for deletion
        word_del_targets = _get_del_targets(word_predictions, tgt_tokens, self.pad)
        word_del_out, _ = self.decoder.forward_word_del(
            normalize=False,
            prev_output_tokens=word_predictions,
            encoder_out=encoder_out)
        word_del_masks = word_predictions.ne(self.pad)

        word_ins_targets = torch.where(tgt_tokens >= self.num_embeddings, torch.ones_like(tgt_tokens) * self.unk_idx, tgt_tokens)
        results = {
            "mask_ins": {
                "out": mask_ins_out, "tgt": mask_ins_targets,
                "mask": mask_ins_masks, "ls": 0.01,
            },
            "word_ins": {
                "out": word_ins_out, "tgt": word_ins_targets,
                "mask": masked_tgt_masks, "ls": self.args.label_smoothing,
                "nll_loss": True
            },
            "word_del": {
                "out": word_del_out, "tgt": word_del_targets,
                "mask": word_del_masks
            }
        }
        return results

    def forward_decoder(
        self, decoder_out, encoder_out, eos_penalty=0.0, delete_penalty=0.0, max_ratio=None,
            no_delete_kwd=False, no_insert_kwd=False, **kwargs
    ):
        kwd_tokens = decoder_out.kwd_tokens
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        history = decoder_out.history

        bsz = output_tokens.size(0)
        if max_ratio is None:
            max_lens = output_tokens.new_zeros(output_tokens.size()[0]).fill_(64)
        else:
            assert False
            if encoder_out.encoder_padding_mask is None:
                max_src_len = encoder_out.encoder_out.size(0)
                src_lens = encoder_out.encoder_out.new(bsz).fill_(max_src_len)
            else:
                src_lens = (~encoder_out.encoder_padding_mask).sum(1)
            max_lens = (src_lens * max_ratio).clamp(min=10).long()

        # delete words
        # do not delete tokens if it is <s> </s>
        can_del_word = output_tokens.ne(self.pad).sum(1) > 2
        kwd_del_masks = _get_kwd_del_masks(
            output_tokens, kwd_tokens, self.bos, self.eos, self.pad, self.sep
        ) if no_delete_kwd else None
        if can_del_word.sum() != 0:  # we cannot delete, skip
            word_del_score, word_del_attn = self.decoder.forward_word_del(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_del_word),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_del_word)
            )
            if delete_penalty > 0.0:
                word_del_score[:, :, 1] = word_del_score[:, :, 1] - delete_penalty
            word_del_pred = word_del_score.max(-1)[1].bool()
            if kwd_del_masks is not None:
                word_del_pred = word_del_pred.masked_fill(kwd_del_masks[can_del_word], 0)

            _tokens, _scores, _attn = _apply_del_words(
                output_tokens[can_del_word],
                output_scores[can_del_word],
                word_del_attn,
                word_del_pred,
                self.pad,
                self.bos,
                self.eos,
            )
            output_tokens = _fill(output_tokens, can_del_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_del_word, _scores, 0)
            attn = _fill(attn, can_del_word, _attn, 0.)

            if history is not None:
                history.append(output_tokens.clone())

        # insert placeholders
        can_ins_mask = output_tokens.ne(self.pad).sum(1) < max_lens
        kwd_ins_masks = _get_kwd_ins_masks(
            output_tokens, kwd_tokens, self.bos, self.eos, self.pad, self.sep
        ) if no_insert_kwd else None
        if can_ins_mask.sum() != 0:
            mask_ins_score, _ = self.decoder.forward_mask_ins(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_ins_mask),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_ins_mask)
            )
            mask_ins_score[:, :, 0] = mask_ins_score[:, :, 0] - eos_penalty
            mask_ins_pred = mask_ins_score.max(-1)[1]
            if kwd_ins_masks is not None:
                mask_ins_pred = mask_ins_pred.masked_fill(kwd_ins_masks[can_ins_mask], 0)
            mask_ins_pred = torch.min(
                mask_ins_pred, max_lens[can_ins_mask, None].expand_as(mask_ins_pred)
            )

            _tokens, _scores = _apply_ins_masks(
                output_tokens[can_ins_mask],
                output_scores[can_ins_mask],
                mask_ins_pred,
                self.pad,
                self.ins,
                self.eos,
            )
            output_tokens = _fill(output_tokens, can_ins_mask, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_mask, _scores, 0)

            if history is not None:
                history.append(output_tokens.clone())

        # insert words
        can_ins_word = output_tokens.eq(self.ins).sum(1) > 0
        if can_ins_word.sum() != 0:
            word_ins_score, word_ins_attn = self.decoder.forward_word_ins(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_ins_word),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_ins_word)
            )
            word_ins_score, word_ins_pred = word_ins_score.max(-1)
            _tokens, _scores = _apply_ins_words(
                output_tokens[can_ins_word],
                output_scores[can_ins_word],
                word_ins_pred,
                word_ins_score,
                self.ins,
            )

            output_tokens = _fill(output_tokens, can_ins_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_word, _scores, 0)
            attn = _fill(attn, can_ins_word, word_ins_attn, 0.)

            if history is not None:
                history.append(output_tokens.clone())

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        attn = None if attn is None else attn[:, :cut_off, :]

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=attn,
            history=history
        )

    def initialize_output_tokens(self, encoder_out, src_tokens, kwd_tokens=None):
        if kwd_tokens is not None:
            max_len = kwd_tokens.size(1)
            initial_output_tokens = kwd_tokens.masked_fill(kwd_tokens==self.sep, self.pad)
            new_orders = utils.new_arange(kwd_tokens).masked_fill(
                kwd_tokens==self.sep, max_len).sort(1)[1]
            initial_output_tokens = initial_output_tokens.gather(1, new_orders)
            initial_output_tokens = initial_output_tokens[:, :initial_output_tokens.
                                                          ne(self.pad).sum(1).max()]
        else:
            initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0), 2)
            initial_output_tokens[:, 0] = self.bos
            initial_output_tokens[:, 1] = self.eos

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out.encoder_out)

        return DecoderOut(
            kwd_tokens=kwd_tokens,
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None
        )


class LevenshteinTransformerDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.pad = dictionary.pad()
        self.ins = dictionary.index('<ins>')
        self.sep = dictionary.index('<sep>')
        self.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
        self.embed_mask_ins = Embedding(256, self.output_embed_dim * 2, None, None)
        self.embed_word_del = Embedding(2, self.output_embed_dim, None, None)

        # del_word, ins_mask, ins_word
        self.early_exit = [int(i) for i in args.early_exit.split(',')]
        assert len(self.early_exit) == 3

        # copy layers for mask-predict/deletion
        self.layers_msk = None
        if getattr(args, "no_share_maskpredictor", False):
            self.layers_msk = nn.ModuleList([
                                TransformerDecoderLayer(args, no_encoder_attn)
                                for _ in range(self.early_exit[1])
                            ])
        self.layers_del = None
        if getattr(args, "no_share_discriminator", False):
            self.layers_del = nn.ModuleList([
                                TransformerDecoderLayer(args, no_encoder_attn)
                                for _ in range(self.early_exit[0])
                            ])

        if getattr(args, "share_discriminator_maskpredictor", False):
            assert getattr(args, "no_share_discriminator", False), "must set saperate discriminator"
            self.layers_msk = self.layers_del

    def extract_features(
        self, prev_output_tokens, encoder_out=None, early_exit=None, layers=None, **unused
    ):
        """
        Similar to *forward* but only return features.
        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        layers = self.layers if layers is None else layers
        early_exit = len(layers) if early_exit is None else early_exit
        for _, layer in enumerate(layers[: early_exit]):
            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
                need_attn=True
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    @ensemble_decoder
    def forward_mask_ins(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out, early_exit=self.early_exit[1], layers=self.layers_msk, **unused
        )
        features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        decoder_out = F.linear(features_cat, self.embed_mask_ins.weight)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra['attn']
        return decoder_out, extra['attn']

    @ensemble_decoder
    def forward_word_ins(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out, early_exit=self.early_exit[2], layers=self.layers, **unused
        )
        decoder_out = self.output_layer(features)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra['attn']
        return decoder_out, extra['attn']

    @ensemble_decoder
    def forward_word_del(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out, early_exit=self.early_exit[0], layers=self.layers_del, **unused
        )
        decoder_out = F.linear(features, self.embed_word_del.weight)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra['attn']
        return decoder_out, extra['attn']


class Embedding(nn.Embedding):
    __constants__ = ["unk_idx"]

    def __init__(self, num_embeddings, embedding_dim, padding_idx, unk_idx):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.unk_idx = unk_idx
        nn.init.normal_(self.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(self.weight[padding_idx], 0)

    def forward(self, input):
        input = torch.where(input >= self.num_embeddings, torch.ones_like(input) * self.unk_idx, input)
        return super().forward(input)


@register_model_architecture("refinement_model", "refinement_base")
def levenshtein_base_architecture(args):
    base_architecture(args)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)
    args.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
    args.early_exit = getattr(args, "early_exit", "6,6,6")
    args.no_share_discriminator = getattr(args, "no_share_discriminator", False)
    args.no_share_maskpredictor = getattr(args, "no_share_maskpredictor", False)
    args.share_discriminator_maskpredictor = getattr(args, "share_discriminator_maskpredictor", False)
    args.no_share_last_layer = getattr(args, "no_share_last_layer", False)

