#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

from collections import Counter
from itertools import zip_longest
import logging
from multiprocessing import Pool
import os
import shutil
import sys

from fairseq import options, tasks, utils
from fairseq.data import indexed_dataset
from fairseq.binarizer import Binarizer


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.preprocess')


def main(args):
    utils.import_user_module(args)

    os.makedirs(args.destdir, exist_ok=True)

    logger.addHandler(logging.FileHandler(
        filename=os.path.join(args.destdir, 'preprocess.log'),
    ))
    logger.info(args)

    task = tasks.get_task(args.task)

    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    def build_dictionary(filenames, field=False, lpos=False, rpos=False, src=False, tgt=False):
        assert field ^ lpos ^ rpos ^ src ^ tgt
        if field:
            return task.build_dictionary(
                filenames,
                workers=args.workers,
                threshold=args.thresholdfield,
                nwords=args.nwordsfield,
                padding_factor=args.padding_factor,
            )
        elif lpos:
            return task.build_dictionary(
                filenames,
                workers=args.workers,
                threshold=args.thresholdlpos,
                nwords=args.nwordslpos,
                padding_factor=args.padding_factor,
            )
        elif rpos:
            return task.build_dictionary(
                filenames,
                workers=args.workers,
                threshold=args.thresholdrpos,
                nwords=args.nwordsrpos,
                padding_factor=args.padding_factor,
            )
        else:
            return task.build_dictionary(
                filenames,
                workers=args.workers,
                threshold=args.thresholdsrc if src else args.thresholdtgt,
                nwords=args.nwordssrc if src else args.nwordstgt,
                padding_factor=args.padding_factor,
            )

    if not args.fielddict and os.path.exists(dict_path(args.field_lang)):
        raise FileExistsError(dict_path(args.field_lang))
    if not args.lposdict and os.path.exists(dict_path(args.lpos_lang)):
        raise FileExistsError(dict_path(args.lpos_lang))
    if not args.rposdict and os.path.exists(dict_path(args.rpos_lang)):
        raise FileExistsError(dict_path(args.rpos_lang))
    if not args.srcdict and os.path.exists(dict_path(args.source_lang)):
        raise FileExistsError(dict_path(args.source_lang))
    if not args.tgtdict and os.path.exists(dict_path(args.target_lang)):
        raise FileExistsError(dict_path(args.target_lang))

    if args.fielddict:
        field_dict = task.load_dictionary(args.fielddict)
    else:
        assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
        field_dict = build_dictionary([train_path(args.field_lang)], field=True)

    if args.lposdict:
        lpos_dict = task.load_dictionary(args.lposdict)
    else:
        assert args.trainpref, "--trainpref must be set if --lposdict is not specified"
        lpos_dict = build_dictionary([train_path(args.lpos_lang)], lpos=True)

    if args.rposdict:
        rpos_dict = task.load_dictionary(args.rposdict)
    else:
        assert args.trainpref, "--trainpref must be set if --rposdict is not specified"
        rpos_dict = build_dictionary([train_path(args.rpos_lang)], rpos=True)

    if args.joined_dictionary:
        assert not args.srcdict or not args.tgtdict, \
            "cannot use both --srcdict and --tgtdict with --joined-dictionary"

        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        elif args.tgtdict:
            src_dict = task.load_dictionary(args.tgtdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary(
                # {train_path(lang) for lang in [args.source_lang, args.target_lang]},
                {train_path(lang) for lang in [args.target_lang]},
                tgt=True
            )
        tgt_dict = src_dict
    else:
        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary([train_path(args.source_lang)], src=True)

        if args.tgtdict:
            tgt_dict = task.load_dictionary(args.tgtdict)
        else:
            assert args.trainpref, "--trainpref must be set if --tgtdict is not specified"
            tgt_dict = build_dictionary([train_path(args.target_lang)], tgt=True)

    field_dict.save(dict_path(args.field_lang))
    lpos_dict.save(dict_path(args.lpos_lang))
    rpos_dict.save(dict_path(args.rpos_lang))
    src_dict.save(dict_path(args.source_lang))
    tgt_dict.save(dict_path(args.target_lang))

    def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers):
        logger.info("[{}] Dictionary: {} types".format(lang, len(vocab)))
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize,
                    (
                        args,
                        input_file,
                        vocab,
                        prefix,
                        lang,
                        offsets[worker_id],
                        offsets[worker_id + 1]
                    ),
                    callback=merge_result
                )
            pool.close()

        ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                          impl=args.dataset_impl, vocab_size=len(vocab))
        merge_result(
            Binarizer.binarize(
                input_file, vocab, lambda t: ds.add_item(t),
                offset=0, end=offsets[1]
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, lang)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))

        logger.info(
            "[{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                vocab.unk_word,
            )
        )

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args.dataset_impl == "raw":
            # Copy original text file to destination folder
            if args.kwd_lang is not None:
                middle_tuple = (args.field_lang, args.lpos_lang, args.rpos_lang, args.source_lang, args.kwd_lang, args.target_lang)
            else:
                middle_tuple = (args.field_lang, args.lpos_lang, args.rpos_lang, args.source_lang, args.target_lang)
            middle = '-'.join(middle_tuple)
            output_text_file = dest_path(output_prefix + "." + middle, lang)
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)
        else:
            make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers)

    def make_all(lang, vocab):
        if args.trainpref:
            make_dataset(vocab, args.trainpref, "train", lang, num_workers=args.workers)
        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(vocab, validpref, outprefix, lang, num_workers=args.workers)
        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(vocab, testpref, outprefix, lang, num_workers=args.workers)

    make_all(args.field_lang, field_dict)
    make_all(args.lpos_lang, lpos_dict)
    make_all(args.rpos_lang, rpos_dict)
    make_all(args.source_lang, src_dict)
    if args.kwd_lang is not None:
        make_all(args.kwd_lang, tgt_dict)
    make_all(args.target_lang, tgt_dict)

    print("| Wrote preprocessed data to {}".format(args.destdir))


def binarize(args, filename, vocab, output_prefix, lang, offset, end, append_eos=True):
    ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                      impl=args.dataset_impl, vocab_size=len(vocab))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, vocab, consumer, append_eos=append_eos,
                             offset=offset, end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args.destdir, output_prefix)
    if args.kwd_lang is not None:
        middle_tuple = (args.field_lang, args.lpos_lang, args.rpos_lang, args.source_lang, args.kwd_lang, args.target_lang)
    else:
        middle_tuple = (args.field_lang, args.lpos_lang, args.rpos_lang, args.source_lang, args.target_lang)
    middle = '-'.join(middle_tuple)
    if lang is not None:
        lang_part = ".{}.{}".format(middle, lang)
    elif args.only_source:
        lang_part = ""
    else:
        lang_part = ".{}".format(middle)

    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def get_offsets(input_file, num_workers):
    return Binarizer.find_offsets(input_file, num_workers)


def cli_main():
    parser = options.get_preprocessing_parser()
    parser.add_argument("-f", "--field-lang", default=None, metavar="FIELD",
                        help="field input")
    parser.add_argument("-l", "--lpos-lang", default=None, metavar="LPOS",
                        help="left position for field")
    parser.add_argument("-r", "--rpos-lang", default=None, metavar="RPOS",
                        help="right position for field")
    parser.add_argument("-k", "--kwd-lang", default=None, metavar="KEYWORD",
                        help="prefix keywords")
    parser.add_argument("--thresholdfield", metavar="N", default=0, type=int,
                        help="map field appearing less than threshold times to unknown")
    parser.add_argument("--thresholdlpos", metavar="N", default=0, type=int,
                        help="map lpos appearing less than threshold times to unknown")
    parser.add_argument("--thresholdrpos", metavar="N", default=0, type=int,
                        help="map rpos appearing less than threshold times to unknown")
    parser.add_argument("--fielddict", metavar="FP",
                        help="reuse given field dictionary")
    parser.add_argument("--lposdict", metavar="FP",
                        help="reuse given lpos dictionary")
    parser.add_argument("--rposdict", metavar="FP",
                        help="reuse given rpos dictionary")
    parser.add_argument("--nwordsfield", metavar="N", default=-1, type=int,
                        help="number of field words to retain")
    parser.add_argument("--nwordslpos", metavar="N", default=-1, type=int,
                        help="number of lpos words to retain")
    parser.add_argument("--nwordsrpos", metavar="N", default=-1, type=int,
                        help="number of rpos words to retain")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
