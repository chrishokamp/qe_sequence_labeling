from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import logging
import json
import os
import re
import cPickle
import itertools
import gzip
import random
import codecs
import datetime
from collections import OrderedDict, defaultdict

from semantic_annotator.datasets import DataProcessor, mkdir_p
from semantic_annotator.spotting import SpacyNERSpotter

import numpy as np
import tensorflow as tf

from tensorflow.contrib import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.contrib.seq2seq.python.ops import attention_decoder_fn
from tensorflow.contrib.seq2seq.python.ops import decoder_fn as decoder_fn_lib
from tensorflow.contrib.seq2seq.python.ops import seq2seq
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.contrib.tensorboard.plugins import projector

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 1.


def directory_iterator(dirname, gzip=False):
    filelist = [os.path.join(dirname, f) for f in os.listdir(dirname)
                if os.path.isfile(os.path.join(dirname, f))]
    # Note: predictable shuffling with random seed?
    random.shuffle(filelist)

    return (r for f in filelist for r in file_iterator(f, gzip=gzip))


def file_iterator(filename, gzip=True):
    """iterator over rows in gzipped file"""
    if gzip:
        with gzip.open(filename) as inp:
            logger.info('Getting Instances from file: {}'.format(filename))
            for r in inp:
                yield r.rstrip().decode('utf8').split('\t')
    else:
        with codecs.open(filename, encoding='utf8') as inp:
            logger.info('Getting Instances from file: {}'.format(filename))
            for r in inp:
                yield r.rstrip().split('\t')


def shuffle_instances_iterator(iterator, shuffle_factor=1000):
    try:
        while True:
            shuffled_instances = [iterator.next() for _ in range(shuffle_factor)]
            random.shuffle(shuffled_instances)
            for ins in shuffled_instances:
                yield ins
    except StopIteration:
        raise StopIteration


def idx_or_unk(tokens, vocab, unknown_token='<UNK>'):
    return [vocab[tok] if tok in vocab else vocab[unknown_token] for tok in tokens]

def padded_batch(seqs, padding_symbol):
    """
    Right pad all seqs in the batch up to the maximum length sequence

    Params:
      seqs: list of sequences
      padding_symbol: the symbol to use for padding

    Returns:
      an np.array with dims (len(seqs), max(len(s) for s in seqs))
    """

    max_len = max(len(s) for s in seqs)
    padded_batch = np.vstack([right_pad(seq, max_len, padding_symbol) for seq in seqs])
    return padded_batch


def mask_batch(batch, mask_symbol):
    mask = np.zeros(batch.shape, dtype='float32')
    # note we assume 2d here
    for i, seq in enumerate(mask):
        if mask_symbol in seq:
            j = seq.index(mask_symbol)
        else:
            j = len(seq)
        mask[i, :j] = 1.
    return mask


def right_pad(seq, max_len, padding_symbol):
    ''' Right pad a sequence up to max_len
    :param seqs:
    :param max_len:
    :return: [seq + padding]
    '''

    # make contexts consistent size
    try:
        padded_context = list(seq) + [padding_symbol for _ in range(max_len - len(seq))]
    except:
        import ipdb; ipdb.set_trace()

    return padded_context


def l2_norm(tensor):
    return tf.sqrt(tf.reduce_sum(tf.square(tensor)))

def load_vocab(vocab_index_file):
    vocab_dict = cPickle.load(open(vocab_index_file))
    vocab_size = len(vocab_dict)
    logger.info('loaded vocabulary index from: {}'.format(vocab_index_file))
    logger.info('vocab size: {}'.format(vocab_size))
    vocab_idict = {v: k for k, v in vocab_dict.items()}
    return vocab_dict, vocab_idict, vocab_size


class UnidirectionalAttentiveQEModel(object):

    def __init__(self, storage, config=None):
        self.storage = storage

        default_config = {
            # model defaults (can be overridden in the .tapm)
            'batch_size': 16,
            'num_steps': 100000,
            'validation_freq': 100,
            'training_transition_cutoff': 50000,
            'max_gradient_norm': 1.0,
            'lstm_stack_size': 2,
            'regularization_alpha': 0.0001,
            'unknown_token': u'<UNK>',
            'learning_rate': 0.1,
            'save_path': 'model.ckpt',
            'sample_prob': 0.5,

            'bos_token': u'<S>',
            'eos_token': u'</S>',
            'encoder_hidden_size': 150,
            'dropout_prob': 0.8,

            'embedding_size': 100,
            'ner_embedding_size': 5,
            'num_ner_tags': 6,
            'ner_padding_token': u'N',

            # TODO: update for QE model
            # 'entity_name_embedding_size': 100,
            # 'entity_bos_id': 1,
            # 'entity_eos_id': 2,
            # 'max_entity_length': 50,
            'decoder_hidden_size': 300
        }

        if config is None:
            config = {}
        self.config = dict(default_config, **config)

        # this is to make results deterministic
        self.random_seed = 42


        # adds factored embeddings for NER tags
        # NER TAG DICT
        self.ner_dict = {u'N': 0, u'B': 1, u'I': 2, u'L': 3, u'O': 4, u'U': 5}
        self.ner_idict = {v: k for k, v in self.ner_dict.items()}
        self.use_ner_embeddings = False

        self.training_log_file = os.path.join(self.storage, 'training_log.out')
        self.validation_records = OrderedDict()

        # WORKING: evaluate with WMT 16 QE data
        # TODO: where there are factors, these need to be segmented as well

        # TODO: configure source and target languages
        # Note: language hardcoding for now
        self.lang = 'en'

        # Note that there is a dependency between the vocabulary index and the embedding matrix
        # TODO: source, target, and output vocab dicts and idicts
        logger.info('Loading vocabulary indices')

        index_dir = self.config['resources']
        src_index = os.path.join(index_dir, 'en.vocab.pkl')
        trg_index = os.path.join(index_dir, 'de.vocab.pkl')
        output_index = os.path.join(index_dir, 'qe_output.vocab.pkl')

        self.src_vocab_dict, self.src_vocab_idict, self.src_vocab_size = load_vocab(src_index)
        self.trg_vocab_dict, self.trg_vocab_idict, self.trg_vocab_size = load_vocab(trg_index)
        self.output_vocab_dict, self.output_vocab_idict, self.output_vocab_size = load_vocab(output_index)

        logger.info('Loading word embeddings')
        # TODO: add pretrained embeddings for target language
        #self.pretrained_embeddings = os.path.join(os.path.dirname(__file__),
        #                                          'resources/embeddings/full_wikipedia.vecs.npz')
        # self.pretrained_embeddings = '/media/1tb_drive/wiki2vec_data/en/text_for_word_vectors/subword_vectors/full_wikipedia.subword.vecs.npz'
        # self.pretrained_source_embeddings = '/media/1tb_drive/wiki2vec_data/en/interesting_sfs/president_king_queen_pm/embeddings/president_king_queen_pm.vecs.npz'
        self.pretrained_source_embeddings = False


        # some of the ops that self._build_graph sets as properties on this instance
        self.graph = None
        self.predictions = None
        self.cost = None
        self.full_graph_optimizer = None
        # self.entity_representation_optimizer = None
        self.saver = None

        logger.info('Building Tensorflow graph')
        self._build_graph()

        self.session = None

    # this function was copied from tensorflow.contrib.seq2seq tests
    @staticmethod
    def _decoder_fn_with_context_state(inner_decoder_fn, name=None):
        """Wraps a given decoder function, adding context state to it.

        Given a valid `inner_decoder_fn`, returns another valid `decoder_fn` which
        first calls `inner_decoder_fn`, then overwrites the context_state, setting
        it to the current time.

        Args:
          inner_decoder_fn: A valid `decoder_fn` of the type passed into
            `dynamic_rnn_decoder`.

        Returns:
          A valid `decoder_fn` to be passed into `dynamic_rnn_decoder`.
        """

        def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
            with ops.name_scope(
                    name, "decoder_fn_with_context_state",
                    [time, cell_state, cell_input, cell_output, context_state]):
                done, next_state, next_input, emit_output, next_context_state = (
                    inner_decoder_fn(time, cell_state, cell_input, cell_output,
                                 context_state))
            next_context_state = time
            return done, next_state, next_input, emit_output, next_context_state

        return decoder_fn


    def _build_graph(self):

        # build the graph
        self.graph = tf.Graph()

        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            # DATASET PLACEHOLDERS

            # (batch, time)
            source = tf.placeholder(tf.int32)
            source_mask = tf.placeholder(tf.float32)
            target = tf.placeholder(tf.int32)
            target_mask = tf.placeholder(tf.float32)
            output = tf.placeholder(tf.int32)
            output_mask = tf.placeholder(tf.float32)

            # TODO: add factored contexts (POS, NER, ETC...)
            # ner_context = tf.placeholder(tf.int32)

            # sets the probability of dropping out
            dropout_prob = tf.placeholder(tf.float32)

            with tf.name_scope('embeddings'):
                source_embeddings = tf.get_variable("source_embeddings",
                                                    [self.src_vocab_size, self.config['embedding_size']],
                                                    trainable=True)
                # TODO: support factors for source and target inputs
                # ner_embeddings = tf.get_variable("ner_embeddings", [self.meta['num_ner_tags'], self.meta['ner_embedding_size']],
                #                                   trainable=True)

                # default: just embed the tokens in the source context
                source_embed = tf.nn.embedding_lookup(source_embeddings, source)

                if self.use_ner_embeddings:
                    pass
                    # TODO: support factors for source input
                    # ner_embed = tf.nn.embedding_lookup(ner_embeddings, ner_context)
                    # context_embed = tf.concat([context_embed, ner_embed], 2)
                    # context_embed.set_shape([None, None, self.meta['embedding_size'] + self.meta['ner_embedding_size']])
                else:
                    # this is to fix shape inference bug in rnn.py -- see this issue: https://github.com/tensorflow/tensorflow/issues/2938
                    source_embed.set_shape([None, None, self.config['embedding_size']])

                # TODO: switch this to target language embeddings
                # TODO: support target language factors (POS, NER, etc...)
                target_embeddings = tf.get_variable("target_embeddings",
                                                    [self.trg_vocab_size, self.config['embedding_size']])

                # target embeddings - these are the _inputs_ to the decoder
                target_embed = tf.nn.embedding_lookup(target_embeddings, target)
                target_embed.set_shape([None, None, self.config['embedding_size']])


            # Construct input representation that we'll put attention over
            # Note: dropout is turned on/off by `dropout_prob`
            with tf.name_scope('input_representation'):
                lstm_cells = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config['encoder_hidden_size'],
                                                                                    use_peepholes=True,
                                                                                    state_is_tuple=True),
                                                            input_keep_prob=dropout_prob)
                              for _ in range(self.config['lstm_stack_size'])]

                cell = tf.contrib.rnn.MultiRNNCell(lstm_cells, state_is_tuple=True)

                # use the description mask to get the sequence lengths
                source_sequence_length = tf.cast(tf.reduce_sum(source_mask, 1), tf.int64)

                # BIDIRECTIONAL RNN
                # Bidir outputs are (output_fw, output_bw)
                bidir_outputs, bidir_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell,
                                                                             inputs=source_embed,
                                                                             sequence_length=source_sequence_length,
                                                                             dtype=tf.float32)
                l_to_r_states, r_to_l_states = bidir_state

                # Transpose to be time-major
                # TODO: do we need to transpose?
                # attention_states = tf.transpose(tf.concat(bidir_outputs, 2), [1, 0, 2])
                attention_states = tf.concat(bidir_outputs, 2)

                # Note: encoder is bidirectional, so we reduce dimensionality by 1/2 to make decoder initial state
                init_state_transformation = tf.get_variable('decoder_init_transform',
                                                            (self.config['encoder_hidden_size']*2,
                                                             self.config['decoder_hidden_size']))
                initialization_state = tf.matmul(tf.concat([r_to_l_states[-1][1], l_to_r_states[-1][1]], 1),
                                                 init_state_transformation)

                # alternatively just use the final l_to_r state
                # initialization_state = l_to_r_states[-1][1]

                # TODO: try with simple L-->R GRU
                # encoder_outputs, encoder_state = rnn.dynamic_rnn(
                #     cell=core_rnn_cell_impl.GRUCell(encoder_hidden_size),
                #     inputs=inputs,
                #     dtype=dtypes.float32,
                #     time_major=False,
                #     scope=scope)

            # Now construct the decoder
            decoder_hidden_size = self.config['decoder_hidden_size']
            # attention
            attention_option = "bahdanau"  # can be "luong"

            with variable_scope.variable_scope("decoder") as scope:

                target_lengths = tf.cast(tf.reduce_sum(target_mask, axis=1), dtype=tf.int32)

                # Prepare attention
                (attention_keys, attention_values, attention_score_fn,
                 attention_construct_fn) = (attention_decoder_fn.prepare_attention(
                    attention_states, attention_option, decoder_hidden_size))

                decoder_fn_train = attention_decoder_fn.attention_decoder_fn_train(
                    encoder_state=initialization_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn)

                # Note: this is different from the "normal" seq2seq encoder-decoder model, because we have different
                # input and output vocabularies for the decoder (target vocab vs. QE symbols)
                num_decoder_symbols = self.output_vocab_size
                # decoder vocab is characters or sub-words? -- either way, we need to learn the vocab over the entity set
                # setting up weights for computing the final output
                # def create_output_fn():
                #     def output_fn(x):
                #         return layers.linear(x, num_decoder_symbols, scope=scope)
                #     return output_fn

                # output_fn = create_output_fn()

                output_transformation = tf.Variable(tf.random_normal([self.config['decoder_hidden_size'], self.output_vocab_size]),
                                                    name='output_transformation')
                output_biases = tf.Variable(tf.zeros([self.output_vocab_size]), name='output_biases')

                # Train decoder
                decoder_cell = core_rnn_cell_impl.GRUCell(decoder_hidden_size)

                (decoder_outputs_train, decoder_state_train, _) = (
                    seq2seq.dynamic_rnn_decoder(
                        cell=decoder_cell,
                        decoder_fn=decoder_fn_train,
                        inputs=target_embed,
                        sequence_length=target_lengths,
                        time_major=False,
                        scope=scope))

                # TODO: for attentive QE, we don't need to separate train and inference decoders
                # TODO: we can directly use train decoder output at both training and prediction time

                # decoder_outputs_train = output_fn(decoder_outputs_train)
                output_shape = tf.shape(decoder_outputs_train)
                decoder_outputs_train = tf.matmul(tf.reshape(decoder_outputs_train,
                                                             [output_shape[0] * output_shape[1], -1]),
                                                  output_transformation)
                decoder_outputs_train += output_biases
                decoder_outputs_train = tf.reshape(decoder_outputs_train, [output_shape[0], output_shape[1], -1])

                # DEBUGGING: dump these
                # self.decoder_outputs_train = decoder_outputs_train

            with tf.name_scope('predictions'):
                prediction_logits = decoder_outputs_train
                logit_histo = tf.summary.histogram('prediction_logits', prediction_logits)

                predictions = tf.nn.softmax(prediction_logits)
                self.predictions = predictions

                # correct_predictions = tf.equal(tf.cast(tf.argmax(predictions, 1), tf.int32), entity)
                # accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
                # accuracy_summary = tf.summary.scalar('accuracy', accuracy)

            with tf.name_scope('xent'):
                # Note: set output and output_mask shape because they're needed here:
                # https://github.com/tensorflow/tensorflow/blob/r1.0/tensorflow/contrib/seq2seq/python/ops/loss.py#L65-L70
                output.set_shape([None, None])
                output_mask.set_shape([None, None])
                costs = tf.contrib.seq2seq.sequence_loss(logits=decoder_outputs_train,
                                                         targets=output,
                                                         weights=output_mask,
                                                         average_across_timesteps=True)
                cost = tf.reduce_mean(costs)
                cost_summary = tf.summary.scalar('minibatch_cost', cost)

            # expose placeholders and ops on the class
            self.source = source
            self.source_mask = source_mask
            self.target = target
            self.target_mask = target_mask
            self.output = output
            self.output_mask = output_mask
            self.predictions = predictions
            self.cost = cost
            self.dropout_prob = dropout_prob

            # TODO: expose embeddings so that they can be visualized?

            optimizer = tf.train.AdamOptimizer()
            with tf.name_scope('train'):
                gradients = optimizer.compute_gradients(cost, tf.trainable_variables())
                if self.config['max_gradient_norm'] is not None:
                    gradients, variables = zip(*gradients)
                    clipped_gradients, _ = clip_ops.clip_by_global_norm(gradients, self.config['max_gradient_norm'])
                    gradients = list(zip(clipped_gradients, variables))

                for gradient, variable in gradients:
                    if isinstance(gradient, ops.IndexedSlices):
                        grad_values = gradient.values
                    else:
                        grad_values = gradient
                    tf.summary.histogram(variable.name, variable)
                    tf.summary.histogram(variable.name + '/gradients', grad_values)
                    tf.summary.histogram(variable.name + '/gradient_norm',
                                         clip_ops.global_norm([grad_values]))

                self.full_graph_optimizer = optimizer.apply_gradients(gradients)

                # Optimizer #2 -- updates entity representations only
                # entity_representation_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                #                                                      "representation/entity_lookup")
                # self.entity_representation_optimizer = optimizer.minimize(cost,
                #                                                           var_list=entity_representation_train_vars)

            self.saver = tf.train.Saver()

            # self.accuracy = accuracy
            self.merged = tf.summary.merge_all()

            logger.info('Finished building model graph')

    def get_batch(self, iterator, batch_size, sample_prob=1.0):
        """

        Params:
          iterator: iterator over (source, target, output) strings
          batch_size: size of the desired batch (note a batch can be smaller if the iterator finishes)
          sample_prob: (optional) the probability of sampling each instance

        Returns:
          (source, source_mask, target, target_mask, output, output_mask)
        """

        i = 0
        sources = []
        targets = []
        outputs = []
        while i < batch_size:
            try:
                source, target, output = iterator.next()
            except StopIteration:
                break

            source_idxs = idx_or_unk(source, self.src_vocab_dict, u'<UNK>')
            target_idxs = idx_or_unk(target, self.trg_vocab_dict, u'<UNK>')
            output_idxs = idx_or_unk(output, self.output_vocab_dict, u'<UNK>')
            assert len(target_idxs) == len(output_idxs), 'Output and target should always be the same length'

            sources.append(source_idxs)
            targets.append(target_idxs)
            outputs.append(output_idxs)

            # reservoir sampling to randomize batches
            if np.random.binomial(1, sample_prob) == 0:
                continue

            i += 1

        src_eos = self.src_vocab_dict[self.config['eos_token']]
        trg_eos = self.trg_vocab_dict[self.config['eos_token']]
        output_eos = self.output_vocab_dict[self.config['eos_token']]

        source_batch = padded_batch(sources, src_eos)
        target_batch = padded_batch(targets, trg_eos)
        output_batch = padded_batch(outputs, output_eos)

        source_mask = mask_batch(source_batch, src_eos)
        target_mask = mask_batch(target_batch, trg_eos)
        output_mask = mask_batch(output_batch, output_eos)

        return source_batch, source_mask, target_batch, target_mask, output_batch, output_mask


    # WORKING HERE: create input iterators for train and dev over (source, target, output) files
    def train(self, train_iter_func, dev_iter_func, restore_from=None, persist_dir=None, logdir=None, auto_log_suffix=True,
              start_iteration=0, shuffle=True):
        """
        Training and dev checks for QE sequence modelsi

        Params:
          training_iter_func: function which returns iterable over (source, mt, labels) instances
          dev_iter_func: function which returns iterable over (source, mt, labels) instances

        """

        if logdir is None:
            logdir = os.path.join(self.storage, 'logs')
        if persist_dir is None:
            persist_dir = os.path.dirname(__file__)

        training_iter = train_iter_func()
        # wrap the data iter to add functionality
        if shuffle:
            training_iter = shuffle_instances_iterator(training_iter, shuffle_factor=5000)

        training_iter = itertools.cycle(training_iter)

        # load pretrained source word embeddings
        source_embeddings = None
        if self.config.get('source_embeddings') is not None:
            source_embeddings = np.load(open(self.config['source_embeddings']))
        # TODO: support pretrained target and output vocabulary embeddings

        if auto_log_suffix:
            log_suffix = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output_logdir = os.path.join(logdir, log_suffix)
        else:
            output_logdir = logdir

        mkdir_p(output_logdir)
        train_writer = tf.summary.FileWriter(output_logdir, self.graph)

        logger.info('Running session, logdir is: {}'.format(output_logdir))
        with tf.Session(graph=self.graph, config=config) as session:

            # Initialization ops
            if restore_from is None:
                tf.initialize_all_variables().run()
                logger.info('Initialized')

                # Pretrained Word Embeddings
                if source_embeddings is not None:
                    session.run(tf.assign(self.word_embeddings, source_embeddings))
                    logger.info('Source word embeddings loaded from: {}'.format(self.config['source_embeddings']))
            else:
                self.saver.restore(session, restore_from)
                logger.info('restored trained model from: {}'.format(restore_from))

            average_loss = 0

            val_freq = self.config['validation_freq']

            # SGD loop
            for step in range(self.config['num_steps']):

                if step % 10 == 0:
                    logger.info('running step: {}'.format(step))

                data_cols = self.get_batch(training_iter,
                                           self.config['batch_size'],
                                           sample_prob=self.config['sample_prob'])

                source, source_mask, target, target_mask, output, output_mask = data_cols

                feed_dict = {
                    self.source: source,
                    self.source_mask: source_mask,
                    self.target: target,
                    self.target_mask: target_mask,
                    self.output: output,
                    self.output_mask: output_mask,
                    self.dropout_prob: self.config['dropout_prob']
                }

                # if step < self.config['training_transition_cutoff']:
                _, l, summary = session.run([self.full_graph_optimizer,
                                            self.cost,
                #                             # self.accuracy,
                                            self.merged], feed_dict=feed_dict)
                # else:
                #     _, l, summary = session.run([self.entity_representation_optimizer,
                #                                            self.cost,
                #                                            # self.accuracy,
                #                                            self.merged], feed_dict=feed_dict)

                train_writer.add_summary(summary, step)

                average_loss += l

                # Validation
                if step % val_freq == 0:
                    logger.info('Running validation...')
                    logger.info('Training loss on last batch: {}'.format(l))

                    dev_iter = dev_iter_func()
                    dev_batch_len = self.config
                    while dev_batch_len > 0:
                        data_cols = self.get_batch(dev_iter,
                                                   dev_batch_len,
                                                   sample_prob=1.0)

                        # this will be zero once the iterator has finished
                        dev_batch_len = len(data_cols[0])
                        if dev_batch_len == 0:
                            continue

                        source, source_mask, target, target_mask, output, output_mask = data_cols

                        feed_dict = {
                            self.source: source,
                            self.source_mask: source_mask,
                            self.target: target,
                            self.target_mask: target_mask,
                            self.output: output,
                            self.output_mask: output_mask,
                            self.dropout_prob: 1.0
                        }

                        preds = session.run(self.predictions, feed_dict=feed_dict)
                        # WORKING HERE: evaluate dev predictions

        logger.info("Step: {} -- Finished Training".format(step))

    def get_entity_embeddings_for_visualization(self, logdir, actual_entities=False):

        ds = tap.dataset.Dataset.load(self.model.dataset)
        # get all dev data
        # Validation set
        test_data = [i for i in ds.collections['test']['examples']]
        test_data_iter = itertools.cycle(self.sequence_model_to_rec_pointer_iterator(ds.collections['test']['examples']))
        len_test = len(test_data)

        # get the entity representations for all of the dev instances
        all_representations = []
        entity_names = []
        for b in range(num_test_batches):
            data_cols, dynamic_map = self.get_batch(test_data_iter,
                                                    self.config['batch_size'],
                                                    sample_prob=1.0,
                                                    smart_sampling=True,
                                                    actual_entities=actual_entities)

            if self.use_ner_embeddings:
                entity, idx_tuple, context, context_mask, ner_context, candidates, candidate_masks = data_cols
            else:
                entity, idx_tuple, context, context_mask, candidates, candidate_masks = data_cols
                ner_context = None

            feed_dict = {
                self.entity: entity,
                self.idx_tuple: idx_tuple,
                self.context: context,
                self.context_mask: context_mask,
                self.ner_context: ner_context,
                self.dropout_prob: 1.0
            }

            entity_representations = self.session.run(self.logits_out, feed_dict=feed_dict)
            all_representations.append(entity_representations)
            entity = [dynamic_map[(i, j)] for i, j in enumerate(entity.flatten())]
            entity_names.extend([self.entity_idict[e] for e in entity])

        all_entity_representations = np.vstack(all_representations)

        # entity_embeddings = session.run(self.entity_weights)
        entity_embedding_var = tf.Variable(all_entity_representations,  name='entity_embeddings')

        self.session.run(entity_embedding_var.initializer)
        summary_writer = tf.summary.FileWriter(logdir)
        entity_embedding_config = projector.ProjectorConfig()
        entity_embedding = entity_embedding_config.embeddings.add()
        entity_embedding.tensor_name = entity_embedding_var.name
        # Comment out if you don't have metadata
        entity_embedding.metadata_path = os.path.join(logdir, 'metadata.tsv')
        projector.visualize_embeddings(summary_writer, entity_embedding_config)
        embedding_saver = tf.train.Saver([entity_embedding_var])
        embedding_saver.save(self.session, os.path.join(logdir, 'entity_embedding.ckpt'), 1)
        entity_type_map = {v: u' '.join(k) for k, values in self.candidate_map.items() for v in values}

        with codecs.open(os.path.join(logdir, 'metadata.tsv'), 'w', encoding='utf8') as emb_metadata:
            emb_metadata.write(u'Name\tSurface_Form\n')
            row_c = 0
            for entity_name in entity_names:
                # entity_name = self.entity_idict.get(entity_idx, u'<UNK>')
                entity_sf = entity_type_map.get(entity_name, u'<UNK>')
                emb_metadata.write('{}\t{}\n'.format(entity_name, entity_sf))
                row_c += 1

        logger.info('Entity embeddings are ready for visualization from: {}'.format(logdir))

    def load(self, model_path):
        """
        load a trained model from a checkpoint file
        """

        if self.session is not None:
            logger.info('Session is already loaded')
            return

        session = tf.Session(graph=self.graph)
        self.saver.restore(session, model_path)
        self.session = session
        logger.info('Restored session from: {}'.format(model_path))

