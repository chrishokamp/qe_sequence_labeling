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
            'context_length': 50, # input context length in words -- currrently this must be >= dataset maximum length
            'batch_size': 2,
            'num_steps': 100000,
            'validation_freq': 100,
            'training_transition_cutoff': 50000,
            'max_gradient_norm': 1.0,
            'embedding_size': 100,
            'lstm_stack_size': 2,
            'regularization_alpha': 0.0001,
            'unknown_token': u'<UNK>',
            'learning_rate': 0.1,
            'save_path': 'model.ckpt',
            'sample_prob': 0.5,

            'bos_token': u'<S>',
            'eos_token': u'</S>',
            'recurrent_layer_size': 150,
            'dropout_prob': 0.8,

            'ner_embedding_size': 5,
            'num_ner_tags': 6,
            'ner_padding_token': u'N',

            # TODO: update for QE model
            # for name-gen seq2seq models
            'entity_name_embedding_size': 100,
            'entity_bos_id': 1,
            'entity_eos_id': 2,
            'max_entity_length': 50,
            'decoder_hidden_size': 300
        }

        if config is None:
            config = {}

        # this is to make results deterministic
        self.random_seed = 42

        self.config = dict(default_config, **config)

        # adds factored embeddings for NER tags
        # NER TAG DICT
        self.ner_dict = {u'N': 0, u'B': 1, u'I': 2, u'L': 3, u'O': 4, u'U': 5}
        self.ner_idict = {v: k for k, v in self.ner_dict.items()}
        self.use_ner_embeddings = False

        self.training_log_file = os.path.join(self.storage, 'training_log.out')
        self.validation_records = OrderedDict()

        # WORKING: evaluate with WMT 16 QE data

        # TODO: segment labels, target, and source into sub-words
        # TODO: where there are factors, these need to be segmented as well
        # self.data_processor = DataProcessor(start_token=self.meta['start_token'],
        #                                     end_token=self.meta['end_token'],
        #                                     max_sequence_length=self.meta['context_length'],
        #                                     keep_start_end_tokens=self.meta['keep_start_end_tokens'],
        #                                     use_subword=True,
        #                                     subword_codes='/media/1tb_drive/wiki2vec_data/en/text_for_word_vectors/subword_encoding/full_wikipedia.en.20000.bpe.codes')

        self.data_processor = DataProcessor(start_token=self.meta['start_token'],
                                            end_token=self.meta['end_token'],
                                            max_sequence_length=self.meta['context_length'],
                                            use_subword=False)


        # TODO: configure source and target languages
        # Note: language hardcoding for now
        self.lang = 'en'

        # Note that there is a dependency between the vocabulary index and the embedding matrix
        # TODO: source, target, and output vocab dicts and idicts
        logger.info('Loading vocabulary index')
        #self.vocabulary_index = os.path.join(os.path.dirname(__file__),
        #                                    'resources/embeddings/full_wikipedia.vocab.pkl')
        # self.vocabulary_index = '/media/1tb_drive/wiki2vec_data/en/text_for_word_vectors/subword_vectors/full_wikipedia.subword.vocab.pkl'
        src_index = '/media/1tb_drive/wiki2vec_data/en/interesting_sfs/president_king_queen_pm/embeddings/president_king_queen_pm.vocab.pkl'
        trg_index = '/media/1tb_drive/wiki2vec_data/en/interesting_sfs/president_king_queen_pm/embeddings/president_king_queen_pm.vocab.pkl'
        output_index = '/media/1tb_drive/wiki2vec_data/en/interesting_sfs/president_king_queen_pm/embeddings/president_king_queen_pm.vocab.pkl'

        self.src_vocab_dict, self.src_vocab_idict, self.src_vocab_size = load_vocab(src_index)
        self.trg_vocab_dict, self.trg_vocab_idict, self.trg_vocab_size = load_vocab(trg_index)
        self.output_vocab_dict, self.output_vocab_idict, self.output_vocab_size = load_vocab(output_index)

        logger.info('Loading word embeddings')
        # TODO: add pretrained embeddings for target language
        #self.pretrained_embeddings = os.path.join(os.path.dirname(__file__),
        #                                          'resources/embeddings/full_wikipedia.vecs.npz')
        # self.pretrained_embeddings = '/media/1tb_drive/wiki2vec_data/en/text_for_word_vectors/subword_vectors/full_wikipedia.subword.vecs.npz'
        self.pretrained_source_embeddings = '/media/1tb_drive/wiki2vec_data/en/interesting_sfs/president_king_queen_pm/embeddings/president_king_queen_pm.vecs.npz'


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
                lstm_cells = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config['recurrent_layer_size'],
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
                                                            self.config['embedding_size']*2,
                                                            self.config['embedding_size'])
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
                def create_output_fn():
                    def output_fn(x):
                        return layers.linear(x, num_decoder_symbols, scope=scope)
                    return output_fn

                output_fn = create_output_fn()

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

                # TODO: add script to create QE output vocab: (target-token-OK, and target-token-BAD)
                decoder_outputs_train = output_fn(decoder_outputs_train)
                # DEBUGGING: dump these
                self.decoder_outputs_train = decoder_outputs_train

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
                if self.meta['max_gradient_norm'] is not None:
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

    # WORKING HERE
    # TODO: update these functions to return (source, source_mask, target, target_mask, output, output_mask)
    def map_data_to_input_matrices(self, entity, idx_tup, context, smart_sampling=False, ner_context=None, num_samples=50):
        """

        :param entity: the unicode entity
        :param idx_tup: iterable w/ 2 int indexes (start, end)
        :param context: iterable of unicode tokens with length strictly <= self.meta['context_length']

        :return: (entity_idx: int, idx_tup: np.array, padded_context_idxs: np.array, context_mask: np.array)
        """

        entity_seq_idxs = np.array(idx_or_unk(entity, self.name_generation_dict, u'<UNK>'))

        max_context_len = self.meta['context_length']
        assert len(context) <= max_context_len, 'you provided a context longer than {} tokens'.format(max_context_len)

        sf = tuple(context[idx_tup[0]:idx_tup[1]])
        try:
            assert sf in self.candidate_map
        except AssertionError:
            logger.error(u'Warning: surface form: {} is not in the candidate map, its disambiguation is: {}'.format(sf, entity))
            return None

        if smart_sampling:
            candidates = self.candidate_map[sf]
        else:
            # Note we're mapping to and from int-->str-->int!?!
            actual_candidates = self.candidate_map[sf]
            num_samples = max(num_samples - len(actual_candidates), 0)
            random_idxs = np.random.choice(self.all_entities_range, size=num_samples, replace=False)
            sampled_candidates = self.all_entities[random_idxs]
            sampled_candidates = list(sampled_candidates) + actual_candidates
            candidates = sampled_candidates

        # Note: some candidates may not be in the entity dict
        # Note: In general, the entity dict and the candidate map should be built from the same dataset
        candidate_idxs = [self.entity_dict[e] for e in candidates if e in self.entity_dict]

        # map contexts to idxs
        eos_idx = self.vocab_dict[self.meta['eos_token']]
        unk_token = self.meta['unknown_token']
        mapped_context = idx_or_unk(context, self.vocab_dict, unk_token)
        padded_context = right_pad(mapped_context, max_context_len, eos_idx)
        training_context = np.array(padded_context)
        actual_length = np.sum(training_context != eos_idx)

        # add a mask where contexts are artificially extended
        context_mask = np.zeros(max_context_len, dtype='float32')
        context_mask[:actual_length] = 1.

        idx_tup = np.array(idx_tup)

        if ner_context is not None:
            # Note: there should never be an unknown token in the ner context
            mapped_ner_context = idx_or_unk(ner_context, self.ner_dict, unk_token)
            ner_eos = self.ner_dict[self.meta['ner_padding_token']]
            padded_ner_context = right_pad(mapped_ner_context, max_context_len, ner_eos)
            training_ner_context = np.array(padded_ner_context)
            return entity_seq_idxs, idx_tup, training_context, context_mask, training_ner_context, candidate_idxs
        else:
            return entity_seq_idxs, idx_tup, training_context, context_mask, candidate_idxs

    def get_batch(self, iterator, batch_size, sample_prob=1.0, smart_sampling=True, actual_entities=False):

        data = []
        i = 0
        while i < batch_size:
            try:
                entity, idx_tup, context = self.parse_recurrent_pointer_fields(iterator.next())
            except StopIteration:
                break

            # WORKING: split entities into characters, lookup the char idxs
            # WORKING: right pad entities in batch

            ner_context = None
            if self.use_ner_embeddings:
                ner_context = self.data_processor.ner_tags_from_pointer_seq_and_context(idx_tup, context)

            # reservoir sampling to randomize batches
            if np.random.binomial(1, sample_prob) == 0:
                continue

            mapped_row = self.map_data_to_input_matrices(entity,
                                                         idx_tup,
                                                         context,
                                                         ner_context=ner_context,
                                                         smart_sampling=smart_sampling)

            # if there was an error in the mapping
            if mapped_row is None:
                continue

            # Note: this filtering makes assumptions about the downstream usage of the model
            # if self.entity_dict[u'<UNK>'] in set(mapped_row[0]):
            #     continue

            # Note: this filtering distorts dev results, because we cannot guarantee that we know the correct entity
            # Note: i.e. the correct entity may not even be present in our index
            # filter out examples where the correct entity is not present in the list of candidates
            if not mapped_row[0][0] in mapped_row[-1]:
                continue

            data.append(mapped_row)
            i += 1

        cols = zip(*data)
        # map entities to their dynamic indices
        #return training_entity, idx_tup, training_context, context_mask, candidate_idxs

        data_cols = cols[1:-1] # idx_tup, training_context, context_mask

        entity_seqs = cols[0]
        entity_lens = [len(e) for e in entity_seqs]
        max_entity_len = max(entity_lens)
        padded_entity_seqs = [right_pad(entity_seq, max_entity_len, self.meta['entity_eos_id'])
                              for entity_seq in entity_seqs]
        entity_mask = np.zeros((batch_size, max_entity_len))
        for i, l in enumerate(entity_lens):
            entity_mask[i, :l] = 1.

        data_cols = [padded_entity_seqs, entity_mask] + data_cols

        output_arrays = [np.vstack(col) for col in data_cols]

        # output_arrays.extend([padded_cands, candidate_mask])

        # return output_arrays, dynamic_candidate_map
        return output_arrays

    def update_model_with_user_data(self, ds, persist=False, persist_dir=None, train_spotter=True):
        # Note: we want them at the character level so that we can dynamically retrain the spotter at the same time as
        # Note: the linking model
        # first pass -- parse user data and update candidate map
        all_training_data = []
        spotter_training_data = []
        i = 0
        for row, y in ds.collection('training'):
            entities, raw_pointer_tups, sfs, text = self.parse_sequence_model_fields(row)

            print('Calling data processor')
            token_pointer_tups = self.data_processor.pointer_seqs_from_spots_and_text(raw_pointer_tups, text)

            assert len(entities) == len(token_pointer_tups), 'We need a training tuple for each entity annotation'
            for entity, (context, pointer_tup) in zip(entities, token_pointer_tups):
                sf = tuple(context[pointer_tup[0]:pointer_tup[1]])
                # update candidate map
                if sf in self.candidate_map:
                    if entity not in self.candidate_map[sf]:
                        if entity in self.entity_dict:
                            # Note: the second check here is to avoid overwriting singleton rules with unknown entities
                            # TODO: remove this once we support retraining with new entities
                            if not (len(self.candidate_map[sf]) == 1 and self.candidate_map[sf][0] not in self.entity_dict):
                                self.candidate_map[sf].append(entity)
                            else:
                                logger.warn(u'I did not add the mapping {}-->{} because {} already points to an ' + \
                                            u'unknown entity: {}'.format(sf,
                                                                         entity,
                                                                         sf,
                                                                         self.candidate_map[sf][0]))
                        else:
                            logger.warn(u'Unknown ambiguous entity in user data: {}'.format(entity))
                    else:
                        logger.info(u'New training instance mapping {}-->{}'.format(sf, entity))
                else:
                    # add hard-mapping rule
                    self.candidate_map[sf] = [entity]
                    if entity not in self.entity_dict:
                        logger.warn(u'New unknown entity in user data: {}, added rule mapping {}-->{}'.format(entity, sf, entity))

                all_training_data.extend([(entity, pointer_tup, context)
                                          for entity, (context, pointer_tup) in zip(entities, token_pointer_tups)])
                spotter_training_data.append([text, raw_pointer_tups])
            print(i)
            i += 1

        print('FINISHED')

        # persist the new candidate map
        cand_map_dir = os.path.join(persist_dir, 'entities/')
        mkdir_p(cand_map_dir)
        cand_map_path = os.path.join(cand_map_dir, 'en.recurrent_pointer.candidate_map.pkl')

        cPickle.dump(self.candidate_map, open(cand_map_path, 'w'))
        logger.info('Saved updated candidate map to: {}'.format(cand_map_path))

        # retrain spotter
        # Note: spotter could be trained async instead of blocking
        if train_spotter:
            spotter_docs, spotter_tups = zip(*spotter_training_data)
            self.update_spotter(spotter_docs, spotter_tups, persist_dir=persist_dir)

        # return iterator over the ambiguous training instances
        for entity, pointer_tup, context in all_training_data:
            sf = tuple(context[pointer_tup[0]:pointer_tup[1]])
            if len(self.candidate_map[sf]) > 1:
                yield (entity, json.dumps(pointer_tup), json.dumps(context))

    def sequence_model_to_rec_pointer_iterator(self, row_iterator):
        for row in row_iterator:
            entities, raw_pointer_tups, sfs, text = self.parse_sequence_model_fields(row)
            token_pointer_tups = self.data_processor.pointer_seqs_from_spots_and_text(raw_pointer_tups, text)
            assert len(entities) == len(token_pointer_tups), 'We need a training tuple for each entity annotation'
            for entity, (context, pointer_tup)in zip(entities, token_pointer_tups):
                yield (entity, json.dumps(pointer_tup), json.dumps(context))

    def train(self, monitor_port, restore_from=None, training_data_dir=None, training_sequence_format='sequence_model',
              persist_dir=None, logdir=None, auto_log_suffix=True,
              train_spotter=True, actual_entities=False):
        """Supports two training modes: one over user data, and one over gzipped files"""

        monitor = tap.classifier.TrainingMonitor(monitor_port)
        ds = tap.dataset.Dataset.load(self.model.dataset)

        if logdir is None:
            logdir = os.path.join(self.storage, 'logs')
        if persist_dir is None:
            persist_dir = os.path.dirname(__file__)

        if training_data_dir is not None:
            assert os.path.isdir(training_data_dir), '`training_data_dir` must point to a directory'
            # hook to get an iterator over all files in dir
            if training_sequence_format == 'sequence_model':
                # WORKING: iterator over instances in sequence model
                data_iter = directory_iterator(training_data_dir, gzip=False)
                data_iter = self.sequence_model_to_rec_pointer_iterator(data_iter)
            elif training_sequence_format != 'recurrent_pointer_model':
                data_iter = directory_iterator(training_data_dir, gzip=True)
            else:
                raise ValueError('Unknown training_sequence_format: {}'.format(training_sequence_format))

            training_data_iter = itertools.cycle(data_iter)

            training_data_iter = shuffle_instances_iterator(training_data_iter, shuffle_factor=5000)
        else:
            # This is the flow for user-provided training data -- optionally updates the spotter as well
            # Updates:
            # (1) the candidate map
            # (2) the spotter(s) -- potentially both probabalistic and string-matching
            # (3) the model -- this should only be for the portion of the training data that is ambiguous
            training_data_iter = self.update_model_with_user_data(ds, persist=True, persist_dir=persist_dir,
                                                                  train_spotter=train_spotter)
            training_data_iter = itertools.cycle(training_data_iter)

        # Validation set
        test_data = [i for i, y in ds.collection('test')]
        test_data_iter = itertools.cycle(self.sequence_model_to_rec_pointer_iterator(test_data))
        len_test = len(test_data)

        # load pretrained word embeddings
        word_embeddings = np.load(open(self.pretrained_embeddings))

        if auto_log_suffix:
            log_suffix = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output_logdir = os.path.join(logdir, log_suffix)
        else:
            output_logdir = logdir

        mkdir_p(output_logdir)
        self.train_writer = tf.summary.FileWriter(output_logdir, self.graph)

        logger.info('Running session, logdir is: {}'.format(output_logdir))
        with tf.Session(graph=self.graph, config=config) as session:

            # Initialization ops
            if restore_from is None:
                tf.initialize_all_variables().run()

                # Pretrained Word Embeddings
                session.run(tf.assign(self.word_embeddings, word_embeddings))
                logger.info('Initialized, word embeddings loaded from: {}'.format(self.pretrained_embeddings))
            else:
                self.saver.restore(session, restore_from)
                logger.info('restored trained model from: {}'.format(restore_from))

            average_loss = 0

            num_test_batches = int(np.ceil(len_test / self.meta['batch_size']))
            val_freq = self.meta['validation_freq']

            # SGD loop
            for step in range(self.meta['num_steps']):

                if step % 10 == 0:
                    logger.info('running step: {}'.format(step))
                    progress = float(step) / float(self.meta['num_steps'])
                    monitor.progress(progress)

                # transitions to only optimizing entity representations after a certain number of iters
                # generate a batch
                #logger.info('about to get batch: {}'.format(step))
                data_cols = self.get_batch(training_data_iter,
                                           self.meta['batch_size'],
                                           sample_prob=self.meta['sample_prob'],
                                           smart_sampling=True,
                                           actual_entities=actual_entities)
                #logger.info('got batch: {}'.format(step))

                if self.use_ner_embeddings:
                    entity_seq, entity_mask, idx_tuple, context, context_mask, ner_context = data_cols
                else:
                    entity_seq, entity_mask, idx_tuple, context, context_mask = data_cols
                    ner_context = None

                feed_dict = {
                    self.entity: entity_seq,
                    self.entity_mask: entity_mask,
                    self.idx_tuple: idx_tuple,
                    self.context: ner_context,
                    self.context_mask: ner_context,
                    self.ner_context: ner_context,
                    self.dropout_prob: self.meta['dropout_prob']
                }
                # print([[self.name_generation_idict[c] for c in seq] for seq in entity_seq])

                if step < self.meta['training_transition_cutoff']:
                    decoder_outputs_train = session.run([self.decoder_outputs_train], feed_dict=feed_dict)
                    _, l, summary = session.run([self.full_graph_optimizer,
                                                self.cost,
                    #                             # self.accuracy,
                                                self.merged], feed_dict=feed_dict)
                else:
                    _, l, summary = session.run([self.entity_representation_optimizer,
                                                           self.cost,
                                                           # self.accuracy,
                                                           self.merged], feed_dict=feed_dict)

                self.train_writer.add_summary(summary, step)

                average_loss += l

                # Validation
                if step % val_freq == 0:
                    logger.info('Running validation...')
                    logger.info('Training loss on last batch: {}'.format(l))

                    test_predictions = []
                    test_entities = []
                    baseline_preds = []

                    per_sf_predictions = defaultdict(list)
                    per_sf_baseline = defaultdict(list)
                    per_sf_ground_truth = defaultdict(list)

                    for b in range(num_test_batches):
                        data_cols = self.get_batch(test_data_iter,
                                                   self.meta['batch_size'],
                                                   sample_prob=1.0,
                                                   smart_sampling=True,
                                                   actual_entities=actual_entities)

                        if self.use_ner_embeddings:
                            entity_seq, entity_mask, idx_tuple, context, context_mask, ner_context = data_cols
                        else:
                            entity_seq, entity_mask, idx_tuple, context, context_mask = data_cols
                            ner_context = None

                        feed_dict = {
                            self.entity: entity_seq,
                            self.entity_mask: entity_mask,
                            self.idx_tuple: idx_tuple,
                            self.context: context,
                            self.context_mask: context_mask,
                            self.ner_context: ner_context,
                            self.dropout_prob: 1.0
                        }

                        preds = session.run(self.predictions, feed_dict=feed_dict)
                        preds = list(np.argmax(preds, axis=2))
                        generated_names = [u''.join([self.name_generation_idict[c] for c in seq]) for seq in preds]
                        actual_names = [u''.join([self.name_generation_idict[c] for c in seq]) for seq in entity_seq]
                        print(zip(generated_names, actual_names))

                        #logger.info('Iter: {} dev-batch: {}, accuracy: {}'.format(step, b, dev_accuracy))

                        # now map back from the dynamic index to the global entity index
                        # we don't need to do this at validation time, just when outputting predictions to the end user
                        # if not actual_entities:
                        #     preds = [dynamic_map[(i,j)] for i,j in enumerate(preds)]
                        #     entity = [dynamic_map[(i,j)] for i,j in enumerate(entity.flatten())]
                        # else:
                        #     entity = entity.flatten()

                        # WORKING: predictions are now sequences of characters
                        # test_predictions.extend(preds)
                        # test_entities.extend(entity)

                        # Test most frequent baselines
                        # y_most_freq_baseline = [self.entity_idict[e_cands[0]] for e_cands in candidates]
                        # baseline_preds.extend(y_most_freq_baseline)

                        # per SF
                        # surface_forms = [tuple(self.vocab_idict[idx] for idx in context_i[pointer_i[0]:pointer_i[1]])
                        #                  for context_i, pointer_i in zip(context, idx_tuple)]
                        # for sf, pred, true, cands in zip(surface_forms, preds, entity, candidates):
                        #     per_sf_predictions[sf].append(pred)
                        #     per_sf_baseline[sf].append(cands[0])
                        #     per_sf_ground_truth[sf].append(true)

                    # map entities back to names
                    # try:
                    #     y_hat = [self.entity_idict[e_idx] for e_idx in test_predictions]
                    #     y_true = [self.entity_idict[e_idx] for e_idx in test_entities]
                    # except:
                    #     import ipdb; ipdb.set_trace()
                    #
                    # test_acc = sum(np.array(y_hat) == np.array(y_true)) / float(len(test_entities))
                    # baseline_acc = sum(np.array(baseline_preds) == np.array(y_true)) / float(len(test_entities))
                    # with codecs.open(self.training_log_file, 'a', encoding='utf8') as logfile:
                    #     logfile.write('ITER: {} TEST_ACC: {}, BASELINE: {}\n'.format(step, test_acc, baseline_acc))
                    #
                    # all_dev_sfs = sorted(per_sf_ground_truth.keys(), key=lambda x: len(per_sf_ground_truth[x]),
                    #                      reverse=True)
                    # per_sf_accs = OrderedDict([(u' '.join(sf),
                    #                            sum(np.array(per_sf_predictions[sf]) == np.array(per_sf_ground_truth[sf]))
                    #                            / float(len(per_sf_ground_truth[sf])))
                    #                            for sf in all_dev_sfs])
                    # per_sf_baseline_accs = OrderedDict([(u' '.join(sf),
                    #                             sum(np.array(per_sf_baseline[sf]) == np.array(per_sf_ground_truth[sf]))
                    #                             / float(len(per_sf_ground_truth[sf])))
                    #                            for sf in all_dev_sfs])
                    #
                    # per_sf_baseline_diffs = OrderedDict([(u' '.join(sf), per_sf_accs[u' '.join(sf)] - per_sf_baseline_accs[u' '.join(sf)])
                    #                                     for sf in all_dev_sfs])
                    #
                    # logger.info("Iter {}: Validation Accuracy on test set: {}".format(step, test_acc))
                    # logger.info("Iter {}: Most freq baseline: {}".format(step, baseline_acc))
                    # logger.info("Iter {}: per SF accs: {}".format(step, json.dumps(per_sf_accs)))
                    # logger.info("Iter {}: per SF baseline accs: {}".format(step, json.dumps(per_sf_baseline_accs)))
                    # logger.info("Iter {}: per SF baseline diffs: {}".format(step, json.dumps(per_sf_baseline_diffs)))
                    # self.validation_records[step] = test_acc
                    # if test_acc == max(v for k,v in self.validation_records.items()):
                    #     save_path = self.saver.save(session, os.path.join(persist_dir, 'model.ckpt'))
                    #     logger.info("Step: {} -- {} is the best score so far, model saved in file: {}".format(step, test_acc, save_path))

        logger.info("Step: {} -- Finished Training".format(step))

    def get_entity_embeddings_for_visualization(self, logdir, actual_entities=False):

        ds = tap.dataset.Dataset.load(self.model.dataset)
        # get all dev data
        # Validation set
        test_data = [i for i in ds.collections['test']['examples']]
        test_data_iter = itertools.cycle(self.sequence_model_to_rec_pointer_iterator(ds.collections['test']['examples']))
        len_test = len(test_data)

        num_test_batches = int(np.ceil(len_test / self.meta['batch_size']))

        # get the entity representations for all of the dev instances
        all_representations = []
        entity_names = []
        for b in range(num_test_batches):
            data_cols, dynamic_map = self.get_batch(test_data_iter,
                                                    self.meta['batch_size'],
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

