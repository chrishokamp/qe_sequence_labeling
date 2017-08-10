import logging
import os
import re
import json
import os
import codecs
from subprocess import Popen, PIPE

from flask import Flask, request, render_template, jsonify, abort

import constrained_decoding

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app = Flask(__name__)
# this needs to be set before we actually run the server
app.models = None


# WORKING: we'll need to call out to TERCOM to get tags
# WORKING: to get confidence, we need TERCOM's predictions for an n-best list of APE outputs
# WORKING: alternatively we would need the softmax output at each timestep
# The question is: given an MT output, how much would you change
# WORKING: this endpoint calls the constrained decoding server after applying pre-/post-processing
@app.route('/word_level_qe', methods=['GET', 'POST'])
def qe_endpoint():
    # TODO: parse request object, remove form
    if request.method == 'POST':
        request_data = request.get_json()
        source_lang = request_data['source_lang']
        target_lang = request_data['target_lang']

        # WORKING: we use the n-best list to compute word-level confidence
        n_best = request_data.get('n_best', 1)
        beam_size = 1
        # return a list of (start, end) indices with labels
        # [
        #     {'start':int, 'end': int, 'label': str, 'confidence': float}
        # ]

        # WORKING: add preprocessing for src+mt concatenation

        if (source_lang, target_lang) not in app.models:
            logger.error('MT Server does not have a model for: {}'.format((source_lang, target_lang)))
            abort(404)

        source_sentence = request_data['source_sentence']

        # WORKING: how to get n-best list from Marian
        # WORKING: all pre-/post-processing for QE here 
        translations = constrained_decoding.server.decode(source_lang, target_lang, source_sentence, n_best=n_best)

    return jsonify({'ranked_translations': translations})


# TODO: add DataProcessor initialized with all the assets we need for pre-/post- processing
# TODO: add terminology mapping hooks into Terminology NMT DataProcessor
# TODO: Map and unmap hooks
# TODO: remember that placeholder term mapping needs to pass the restore map through to postprocessing
# TODO: restoring @num@ placeholders with word alignments? -- leave this for last


def decode(source_lang, target_lang, source_sentence, constraints=None, n_best=1, length_factor=1.5, beam_size=5):
    """
    Decode an input sentence

    Args:
      source_lang: two char src lang abbreviation
      target_lang: two char src lang abbreviation
      source_sentence: the source sentence to translate (we assume already preprocessed)
      n_best: the length of the n-best list to return (default=1)

    Returns:

    """

    model = app.models[(source_lang, target_lang)]
    decoder = app.decoders[(source_lang, target_lang)]
    # Note: remember we support multiple inputs for each model (i.e. each model may be an ensemble where sub-models
    # accept different inputs)

    data_processor = app.processors.get((source_lang, target_lang), None)
    if data_processor is not None:
        source_sentence = u' '.join(data_processor.tokenize(source_sentence))

    inputs = [source_sentence]

    mapped_inputs = model.map_inputs(inputs)

    input_constraints = []
    if constraints is not None:
        input_constraints = model.map_constraints(constraints)

    start_hyp = model.start_hypothesis(mapped_inputs, input_constraints)

    beam_size = max(n_best, beam_size)
    search_grid = decoder.search(start_hyp=start_hyp, constraints=input_constraints,
                                 max_hyp_len=int(round(len(mapped_inputs[0][0]) * length_factor)),
                                 beam_size=beam_size)

    best_output, best_alignments = decoder.best_n(search_grid, model.eos_token, n_best=n_best,
                                                  return_model_scores=False, return_alignments=True,
                                                  length_normalization=True)

    if n_best > 1:
        # start from idx 1 to cut off `None` at the beginning of the sequence
        # separate each n-best list with newline
        decoder_output = [u' '.join(s[0][1:]) for s in best_output]
    else:
        # start from idx 1 to cut off `None` at the beginning of the sequence
        decoder_output = [u' '.join(best_output[0][1:])]

    return decoder_output


def run_ape_qe_server(processors, port=5007):
    # Caller passes in a dict of DataProcessors, keys are language codes
    app.processors = processors

    logger.info('Server starting on port: {}'.format(port))
    # logger.info('navigate to: http://localhost:{}/neural_MT_demo to see the system demo'.format(port))
    app.run(debug=True, port=port, host='127.0.0.1', threaded=True)


# Note: this function will break libgpuarray if theano is using the GPU
# def run_imt_server(models, processors=None, port=5007):
#     # Note: servers use a special .yaml config format-- maps language pairs to NMT configuration files
#     # the server instantiates a predictor for each config, and hashes them by language pair tuples -- i.e. (en,fr)
#     # Caller passes in a dict of predictors, keys are tuples (source_lang, target_lang)
#     if processors is None:
#         app.processors = {k: None for k in models.keys()}
#     else:
#         app.processors = processors
#
#     app.models = models
#     app.decoders = {k: create_constrained_decoder(v) for k, v in models.items()}
#
#
#     logger.info('Server starting on port: {}'.format(port))
#     # logger.info('navigate to: http://localhost:{}/neural_MT_demo to see the system demo'.format(port))
#     # app.run(debug=True, port=port, host='127.0.0.1', threaded=True)
#     app.run(debug=True, port=port, host='127.0.0.1', threaded=False)

