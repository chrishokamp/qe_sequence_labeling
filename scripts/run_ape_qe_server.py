# coding: utf-8

import json
import codecs
import argparse

from sequence_qe.server import DataProcessor
from sequence_qe.server.ape_qe_app import run_ape_qe_server


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()

    # Note: using only one source subword dict stops us from using multiple models with different inputs
    parser.add_argument('--source_subword_codes', help='path to the source subword codes', required=True)
    parser.add_argument('--target_subword_codes', help='path to the target subword codes', required=True)

    parser.add_argument('--source_lang', type=str, help='two-character source language code', required=True)
    parser.add_argument('--target_lang', type=str, help='two-character target language code', required=True)
    parser.add_argument('--port', type=int, help='the port where the server will be exposed', required=False,
                        default=5007)
    parser.add_argument('--marian_port', type=int, help='the port on localhost where the Marian server is listening',
                        required=False,
                        default=8080)
    parser.add_argument('--source_truecase', default=None, type=str,
                        help='(Optional) Path to the source truecasing model',
                        required=False)
    parser.add_argument('--target_truecase', default=None, type=str,
                        help='(Optional) Path to the target truecasing model',
                        required=False)
    parser.add_argument('--escape_special_chars', dest='escape_special_chars', action='store_true',
                        help='(Optional) if --escape_special_chars, we will map special punctuation to html entities')
    parser.set_defaults(escape_special_chars=False)
    args = parser.parse_args()

    assert len(args.source_lang) == len(args.target_lang) == 2, 'Language codes must be two characters'

    # Make a data processor for this model
    # Note: we need different processors for every possible source and target language
    src_data_processor = DataProcessor(lang=args.source_lang, use_subword=True,
                                       subword_codes=args.source_subword_codes,
                                       truecase_model=args.source_truecase,
                                       escape_special_chars=args.escape_special_chars)
    trg_data_processor = DataProcessor(lang=args.target_lang, use_subword=True,
                                       subword_codes=args.target_subword_codes,
                                       truecase_model=args.target_truecase,
                                       escape_special_chars=args.escape_special_chars)

    processor_dict = {
        args.source_lang: src_data_processor,
        args.target_lang: trg_data_processor
    }

    run_ape_qe_server(processors=processor_dict, port=args.port, marian_port=args.marian_port)




