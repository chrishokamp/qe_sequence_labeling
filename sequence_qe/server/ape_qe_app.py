import logging
import os
import re
import json
import os
import codecs
from subprocess import Popen, PIPE

from sequence_qe import dataset

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
@app.route('/word_level_qe', methods=['GET', 'POST'])
def qe_endpoint():
    if request.method == 'POST':
        request_data = request.get_json()
        source_lang = request_data['src_lang']
        target_lang = request_data['trg_lang']

        # WORKING: we use the n-best list to compute word-level confidence
        # return a list of (start, end) indices with labels
        # [
        #     {'start':int, 'end': int, 'label': str, 'confidence': float}
        # ]

        # if (source_lang, target_lang) not in app.models:
        #     logger.error('MT Server does not have a model for: {}'.format((source_lang, target_lang)))
        #     abort(404)

        raw_source_sentence = request_data['src_segment']
        raw_target_sentence = request_data['trg_segment']

        # Note: dependency between input preprocessing and model server
        source_sentence = preprocess(source_lang, raw_source_sentence)
        target_sentence = preprocess(target_lang, raw_target_sentence)

        model_input = concat_src_trg(source_sentence, target_sentence)
        print(model_input)

        translations = decode(model_input, port=app.marian_port)
        print(translations)

        # note that we just fully post process the translations, and use the _untokenized_ target input here,
        # note this could be suboptimal
        # note this method saves some complex span re-alignment logic
        translations = [postprocess(target_lang, trans) for trans in translations]
        hyps = [raw_target_sentence] * len(translations)

        # now get the TER alignments for the n-best translations
        qe_labels = dataset.extract_ter_alignment(hyps,
                                                  translations,
                                                  source_lang,
                                                  target_lang,
                                                  app.tercom_path)
        tag_stacks = zip(*qe_labels)
        ok_counts = [sum([1 for tag in tag_stack if tag == 'OK']) for tag_stack in tag_stacks]
        # note directly splitting input MT sequence on whitespace
        nbest_size = float(len(translations))
        ok_prob = [c / nbest_size for c in ok_counts]
        bad_prob = [1.0 - p for p in ok_prob]
        confidence_dict = {'OK': ok_prob, 'BAD': bad_prob}

        # we return the 1-best tags, but use the n-best list to get confidence
        # TODO: now map all tags to character-level spans
        qe_objs = []
        for idx, tag in enumerate(qe_labels[0]):
            qe_obj = {
                'tag': tag,
                'confidence': confidence_dict[tag][idx],
                'idx': idx
            }
            qe_objs.append(qe_obj)

    return jsonify({'qe_labels': qe_objs})


def decode(source_text, domain='ws://localhost', port='8080'):
    print("DECODE")

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

