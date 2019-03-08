# -*- coding: utf-8 -*-
from __future__ import absolute_import

import argparse
import gc

import logging
import os
import random
import sys

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

logger = logging.getLogger()


def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=40,
                        help='embedding size')
    parser.add_argument('--max_seq_length', type=int, default=1000,
                        help='max sequence length')
    parser.add_argument('--max_nb_words', type=int, default=20000,
                        help='max number of words in document')
    parser.add_argument('--dataset_name',
                        help='dataset name')
    parser.add_argument('--root_dir',
                        help='root directory')
    parser.add_argument('--text_data_dir',
                        help='text data dir')
    parser.add_argument('--word_vec_dir',
                        help='word vector directory')
    parser.add_argument('--query_strategy', nargs='?',
                        help='data point query strategy: Random, Uncertain, Diversity')
    parser.add_argument('--episodes', type=int,
                        help='number of training episodes')
    parser.add_argument('--ndream', type=int, default=5,
                        help='number of dream in dreaming phase')
    parser.add_argument('--timesteps', type=int, default=10,
                        help='number of training timesteps in a episode')
    parser.add_argument('--k', type=int, default=5,
                        help='k - number of samples to query each time')
    parser.add_argument('--annotation_budget', type=int, default=200,
                        help='annotation budget')
    parser.add_argument('--dreaming_budget', type=int, default=10,
                        help='dreaming budget')
    parser.add_argument('--output',
                        help='Output folder')
    parser.add_argument('--test_set', nargs='?',
                        help='Test set path')
    parser.add_argument('--validation_size', type=int, default=10,
                        help='Number of data to leave out for validation in each episode')
    parser.add_argument('--label_data_size', type=int, default=200,
                        help='Number of labeled data. The rest will be treated as unlabel data in each episode')
    parser.add_argument('--initial_training_size', type=int, default=5,
                        help='Number of data point to initialize underlying model in dreaming phase')
    parser.add_argument('--policy_path', nargs='?',
                        help='policy path')
    parser.add_argument('--model_path', nargs='?',
                        help='model path')
    parser.add_argument('--learning_phase_length', type=int, default=10,
                        help='number of datapoint to get annotation on ')
    parser.add_argument('--k_fold', type=int, default=10,
                        help='k - fold cross validation')
    parser.add_argument('--k_learning', type=int, default=5,
                        help='k value in active learning phase')
    parser.add_argument('--n_learning', type=int, default=100,
                        help='top n uncertainty for candidate selection in active learning phase')
    parser.add_argument('--classifier_batch_size', type=int, default=1,
                        help='batch size for trainning classifier')
    parser.add_argument('--classifier_epochs', type=int, default=1,
                        help='num epochs for trainning classifier')
    parser.add_argument('--classifier_learning_rate', type=float, default=0.0005,
                        help='classifier learning rate')
    parser.add_argument('--dreaming_candidate_selection_mode', default="random",
                        help='How to select candidate for dreaming: random, certainty, mix')
    parser.add_argument('--al_candidate_selection_mode', default="random",
                        help='How to select candidate in AL phase: random, uncertainty')

    return parser.parse_args()

def load_embeddings(embedding_path):
    logger.info('Indexing word vectors.')
    embeddings_index = {}
    # if sys.version_info < (3,):
    #     f = open(embedding_path)
    # else:
    #     f = open(embedding_path, encoding='latin-1')
    f = open(embedding_path)
    for line in f:
        values = line.split()
        word = values[0]
        # logger.info(line)
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    logger.info('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def load_data(data_dir, max_nb_words, max_seq_length):
    logger.info('Processing text dataset')
    texts = []  # list of text samples
    labels_index = {}  # # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in os.listdir(path):
                if len(fname) > 0:
                    fpath = os.path.join(path, fname)

                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    t = f.read().rstrip().strip().replace('\n', ' ').replace('\t', '')
                    t = t.replace('1s', '').replace('1n', '')
                    t = t.lower()
                    texts.append(t)
                    f.close()
                    labels.append(label_id)
    logger.info('Found %s texts' % len(texts))

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=max_nb_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True,
                          split=" ",
                          char_level=False)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    logger.info('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=max_seq_length)
    labels = to_categorical(np.asarray(labels))
    logger.debug('Shape of data tensor: {}'.format(str(data.shape)))
    logger.debug('Shape of label tensor: {}'.format(str(labels.shape)))
    del texts
    gc.collect()
    return data, labels, word_index

def construct_embedding_table(embeddings_index, word_index, max_nb_words, embedding_dim):
    logger.info('Preparing embedding matrix.')
    # prepare embedding matrix
    num_words = max(max_nb_words, len(word_index))
    embedding_matrix = np.zeros((num_words + 1, embedding_dim))
    for word, i in word_index.items():
        if i > max_nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix, num_words

def partition_data(data, labels, size, shuffle=True):
    zipdata = list(zip(data, labels))
    if shuffle:
        logger.info('Shuffle data before partition')
        random.shuffle(zipdata)
    sample_idxs = random.sample(range(len(zipdata)), size)
    samples = [zipdata[i] for i in sample_idxs]
    new_data = []
    for idx in range(len(zipdata)):
        if idx not in sample_idxs:
            new_data.append(zipdata[idx])
    if size > 0:
        x_la, y_la = [list(t) for t in zip(*samples)]
    else:
        x_la = []
        y_la = []
    x_un, y_un = [list(t) for t in zip(*new_data)]
    return np.asarray(x_la), np.asarray(y_la), np.asarray(x_un), np.asarray(y_un)

def shuffle_test_data(data, labels):
    logger.info('Shuffle data')
    indices = np.arange(len(data))
    np.random.seed(43)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    return data, labels

def partition_test_data(data, labels, k_fold, idx):
    logger.info("Partition test data to {} folds - fold {}".format(k_fold, idx))
    if idx >= k_fold:
        raise("Invalid idx_fold_xval values. index larger than number of fold")
    num = k_fold * 2
    size = int(len(data) / num)
    x_dim = np.shape(data)[1]
    y_dim = np.shape(labels)[1]
    data = data.reshape(num, size, x_dim)
    labels = labels.reshape(num, size, y_dim)
    x_la = np.vstack([data[idx], data[num-idx-1]])
    y_la = np.vstack([labels[idx], labels[num-idx-1]])
    data = np.delete(data, (idx, num-idx-1) , axis=0)
    labels = np.delete(labels, (idx, num-idx-1), axis=0)
    x_val = data.reshape(-1, x_dim)
    y_val = labels.reshape(-1, y_dim)
    return x_la, y_la, x_val, y_val