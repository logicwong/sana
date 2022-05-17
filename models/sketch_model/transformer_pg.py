# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.functional as F

from fairseq import utils, metrics
from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairseq.models.fairseq_encoder import EncoderOut
from models.data2text_transformer import (
    TransformerModel,
    TransformerDecoder,
    base_architecture,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    Linear
)

from torch import Tensor


logger = logging.getLogger(__name__)


@register_model('sketch_model')
class TransformerPointerGeneratorModel(TransformerModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        TransformerModel.add_args(parser)
        parser.add_argument('--source-position-markers', type=int, metavar='N',
                            help='dictionary includes N additional items that '
                                 'represent an OOV token at a particular input '
                                 'position')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

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
        return TransformerPointerGeneratorDecoder(
            args,
            tgt_dict,
            embed_tokens
        )


class TransformerPointerGeneratorDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`. The pointer-generator variant mixes
    the output probabilities with an attention distribution in the output layer.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False)

        # The dictionary may include a separate entry for an OOV token in each
        # input position, so that their identity can be restored from the
        # original source text.
        self.num_types = len(dictionary)
        self.num_oov_types = args.source_position_markers
        self.num_embeddings = self.num_types - self.num_oov_types

        self.Q = Linear(512, 256)
        self.K = Linear(512, 256)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = 0,
        alignment_heads: Optional[int] = 1,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (EncoderOut, optional): output from the encoder, used
                for encoder-side attention
            incremental_state (dict, optional): dictionary used for storing
                state during :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False)
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state
        )
        attn_weights = torch.bmm(self.Q(x), self.K(encoder_out.encoder_out).permute(1, 2, 0))
        attn_weights = attn_weights.masked_fill(encoder_out.encoder_padding_mask[:, None, :], float("-inf"))
        attn = torch.softmax(attn_weights / 16, dim=-1)

        if not features_only:
            x = self.output_layer(x, attn, encoder_out.src_tokens)
        return x, extra

    def output_layer(self, features, attn, src_tokens, **kwargs):
        batch_size = src_tokens.shape[0]
        src_length = src_tokens.shape[1]
        output_length = features.shape[1]

        # Scatter attention distributions to distributions over the extended
        # vocabulary in a tensor of shape [batch_size, output_length,
        # vocab_size]. Each attention weight will be written into a location
        # that is for other dimensions the same as in the index tensor, but for
        # the third dimension it's the value of the index tensor (the token ID).
        index = src_tokens[:, None, :]
        index = index.expand(batch_size, output_length, src_length)
        attn_dists_size = (batch_size, output_length, self.num_types)
        attn_dists = attn.new_zeros(attn_dists_size)
        attn_dists.scatter_add_(2, index, attn)

        # Final distributions, [batch_size, output_length, num_types].
        return attn_dists

    def get_normalized_probs(self, net_output, log_probs, sample):
        """
        Get normalized probabilities (or log probs) from a net's output.
        Pointer-generator network output is already normalized.
        """
        probs = net_output[0]
        # Make sure the probabilities are greater than zero when returning log
        # probabilities.
        return probs.clamp(1e-10, 1.0).log() if log_probs else probs


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


@register_model_architecture("sketch_model", "sketch_base")
def transformer_pointer_generator(args):
    base_architecture(args)