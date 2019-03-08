# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:03:05 2017

@author: lming
"""
import time

import utils
from model import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

start_time = time.time()
args = utils.get_args()
logger = utils.init_logger()

EMBEDDING_DIM = args.embedding_dim
MAX_SEQUENCE_LENGTH = args.max_seq_length
MAX_NB_WORDS = args.max_nb_words

rootdir = args.root_dir
DATASET_NAME = args.dataset_name
TEXT_DATA_DIR = args.text_data_dir
TEST_DIR = args.test_set
GLOVE_DIR = args.word_vec_dir
BATCH_SIZE = 32
EPOCHS = 30
classifiername="{}/{}_classifier.h5".format(args.output, DATASET_NAME)

logger.info("Train classifier on dataset {}".format( DATASET_NAME))
logger.info(" * INPUT directory: {}".format(TEXT_DATA_DIR))
logger.info(" * OUTPUT classfier {}".format(classifiername))

# first, build index mapping words in the embeddings set
# to their embedding vector
embeddings_index = utils.load_embeddings(GLOVE_DIR)

# second, prepare text samples and their labels
data, labels, word_index = utils.load_data(TEXT_DATA_DIR, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
# test_data, test_labels, _ = utils.load_data(TEST_DIR, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
# data set for inisialize the model
test_data, test_labels, train_data, train_labels = utils.partition_data(data, labels, 100, shuffle=True)
dev_data, dev_labels, train_data, train_labels = utils.partition_data(train_data, train_labels, 5, shuffle=True)
logger.info("Dataset size: train = {}, test = {}, dev = {}".format(len(train_data), len(test_data), len(dev_data)))
embedding_matrix, num_words = utils.construct_embedding_table(embeddings_index, word_index, MAX_NB_WORDS, EMBEDDING_DIM)

logger.info('Set TF configuration')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
config.log_device_placement = False
set_session(tf.Session(config=config))

logger.info('Begin train classifier..')
model = getClassifier(num_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH)
model.fit(train_data, train_labels, validation_data=(dev_data, dev_labels), batch_size=BATCH_SIZE, epochs=EPOCHS)
accuracy = model.evaluate(test_data, test_labels, verbose=0)[2]
logger.info("Accurary : {}".format( str(accuracy)))
model.save(classifiername)
logger.info("Training complete")