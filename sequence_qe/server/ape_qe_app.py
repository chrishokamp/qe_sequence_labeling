import logging
import os
import re
import json
import os
import codecs
from subprocess import Popen, PIPE

from sequence_qe import dataset
from sequence_qe import server

from flask import Flask, request, render_template, jsonify, abort
from websocket import create_connection

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app = Flask(__name__)
# this needs to be set before we actually run the server
app.models = None


def concat_src_trg(src, trg, break_token=u'@BREAK@'):
    return u'{} {} {}'.format(src, break_token, trg)


# WORKING: all preprocessing here (for both src and trg seq)
def preprocess(lang, text):
    assert type(text) is unicode, 'preprocessing only accepts unicode input'
    if lang not in app.processors:
        logger.error('MT Server does not have a DataProcessor for: {}'.format(lang))
        raise ValueError

    data_processor = app.processors[lang]

    # preprocess
    # Note: we may need to factor out the tokenization steps and run them individually
    text = data_processor.tokenize(text)

    return text


def postprocess(lang, text):
    # note we need to exactly reverse what happened in `preprocess`, but this must be done step-by-step so that we don't screw up the span indices
    assert type(text) is unicode, 'postprocessing only accepts unicode input'
    if lang not in app.processors:
        logger.error('MT Server does not have a DataProcessor for: {}'.format(lang))
        raise ValueError

    data_processor = app.processors[lang]

    # postprocess
    # Note: we may need to factor out the tokenization steps and run them individually
    text = data_processor.detokenize(text)
    text = data_processor.deescape_special_chars(text)
    text = data_processor.detruecase(text)
    return text


# WORKING: we'll need to call out to TERCOM to get tags
# WORKING: to get confidence, we need TERCOM's predictions for an n-best list of APE outputs
# WORKING: alternatively we would need the softmax output at each timestep
# The question is: given an MT output, how much would you change
# WORKING: this endpoint calls the constrained decoding server after applying pre-/post-processing
# TODO: persistent cache on this server
@app.route('/word_level_qe', methods=['GET', 'POST'])
def qe_endpoint():
    if request.method == 'POST':
        request_data = request.get_json()
        source_lang = request_data['src_lang']
        target_lang = request_data['trg_lang']

        # if (source_lang, target_lang) not in app.models:
        #     logger.error('MT Server does not have a model for: {}'.format((source_lang, target_lang)))
        #     abort(404)

        raw_source_sentence = request_data['src_segment']
        raw_target_sentence = request_data['trg_segment']

        # Note: dependency between input preprocessing and model server
        source_sentence = preprocess(source_lang, raw_source_sentence)
        target_sentence = preprocess(target_lang, raw_target_sentence)

        model_input = concat_src_trg(source_sentence, target_sentence)
        raw_translations = decode(model_input, port=app.marian_port)

        # note that we just fully post process the translations, and use the _untokenized_ target input here,
        # note this could be suboptimal
        # note this method saves some complex span re-alignment logic
        # TODO: map to spans before or after postprocessing?

        target_data_processor = app.processors.get(target_lang, None)
        # this is a hack to make sure escaped punctuation gets matched correctly
        if target_data_processor.escape_special_chars:
            no_specials = []
            for output in raw_translations:
                no_special = u' '.join([target_data_processor.deescape_special_chars(tok) for tok in output.split()])
                no_specials.append(no_special)

            target_sentence = u' '.join([target_data_processor.deescape_special_chars(tok)
                                         for tok in target_sentence.split()])
            translations = no_specials


        hyps = [target_sentence] * len(translations)

        # translations = [postprocess(target_lang, trans) for trans in raw_translations]

        # Note: what we pass in as `hyps` dictates how many output tags there will be
        # now get the TER alignments for the n-best translations
        qe_labels = dataset.extract_ter_alignment(hyps,
                                                  translations,
                                                  source_lang,
                                                  target_lang,
                                                  app.tercom_path)

        tag_stacks = zip(*qe_labels)
        ok_counts = [sum([1 for tag in tag_stack if tag == 'OK']) for tag_stack in tag_stacks]

        nbest_size = float(len(translations))
        ok_prob = [c / nbest_size for c in ok_counts]
        bad_prob = [1.0 - p for p in ok_prob]
        confidence_dict = {'OK': ok_prob, 'BAD': bad_prob}

        # we return the 1-best tags, but use the n-best list to get confidence
        # TODO: now map all tags to character-level spans
        hyp_tokens = hyps[0].split()
        hyp_token_spans = []
        current_prefix = u''
        for token in hyp_tokens:
            if len(current_prefix) == 0:
                start_idx = len(current_prefix)
                current_prefix = token
            else:
                start_idx = len(current_prefix) + 1
                current_prefix = u'{} {}'.format(current_prefix, token)
            end_idx = len(current_prefix)
            hyp_token_spans.append((start_idx, end_idx))

        # now remap the spans
        if target_data_processor.truecase:
            # just a hack to make sure the capitalization of the mapped target and the original target matches
            target_sentence = raw_target_sentence[0] + target_sentence[1:]
            # target_sentence = target_data_processor.detruecase(target_sentence)
        hyp_token_spans = server.remap_constraint_indices(target_sentence, raw_target_sentence, hyp_token_spans)

        # for idx, span in enumerate(hyp_token_spans):
        #     print(u'span: {}, sf: {}, tag: {}, conf: {}'.format(span,
        #                                                         raw_target_sentence[span[0]:span[1]],
        #                                                         qe_labels[0][idx],
        #                                                         confidence_dict[qe_labels[0][idx]][idx]))

        qe_objs = []
        for idx, tag in enumerate(qe_labels[0]):
            qe_obj = {
                'tag': tag,
                'confidence': confidence_dict[tag][idx],
                'span': hyp_token_spans[idx]
            }
            qe_objs.append(qe_obj)

    return jsonify({'qe_labels': qe_objs})


def decode(source_text, domain='ws://localhost', port='8080'):
    ws_location = '{}:{}/translate'.format(domain, port)
    # Note: this way of using the websocket is wasteful, because we don't take advantage of the ability to keep it open
    ws = create_connection(ws_location)

    # build the request object
    # we wrap input in a list because Marian server expects lists
    message = json.dumps({
        'segments': [source_text],
    })

    ws.send(message)
    decoded_segments = json.loads(ws.recv())
    ws.close()

    return decoded_segments['segments']


def run_ape_qe_server(processors, port=5007, marian_port=8080):

    # Caller passes in a dict of DataProcessors, keys are language codes
    app.processors = processors

    # the port where the Marian websocket is open
    app.marian_port = marian_port

    app.tercom_path = os.path.join(os.path.dirname(__file__), 'resources/ter')

    logger.info('Server starting on port: {}, interfacing with Marian on port: {}'.format(port, marian_port))
    app.run(debug=True, port=port, host='127.0.0.1', threaded=True)

