# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:40:41 2017

@author: lming
"""
import gc

from queryStrategy import *
from model import *
import utils
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

args = utils.get_args()
logger = utils.init_logger()

EMBEDDING_DIM = args.embedding_dim
MAX_SEQUENCE_LENGTH = args.max_seq_length
MAX_NB_WORDS = args.max_nb_words

rootdir=args.root_dir
DATASET_NAME=args.dataset_name
TEXT_DATA_DIR = args.text_data_dir
GLOVE_DIR = args.word_vec_dir

QUERY=args.query_strategy
EPISODES=args.episodes
BUDGET=args.annotation_budget
numofsamples=1
TEST_DIR = args.test_set
resultname= "{}/{}_accuracy.txt".format(args.output, DATASET_NAME)
logger.info("Run AL baseline [{}] on dataset {}".format(QUERY, DATASET_NAME))
logger.info(" * INPUT directory: {}".format(TEXT_DATA_DIR))
logger.info(" * OUTPUT file: {}".format(resultname))

# first, build index mapping words in the embeddings set
# to their embedding vector
embeddings_index = utils.load_embeddings(GLOVE_DIR)

# second, prepare text samples and their labels
data, labels, word_index = utils.load_data(TEXT_DATA_DIR, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
# test_data, test_labels, _ = utils.load_data(TEST_DIR, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
# test_data, test_labels = utils.shuffle_test_data(test_data, test_labels)

x_un, y_un = data, labels
embedding_matrix, num_words = utils.construct_embedding_table(embeddings_index, word_index, MAX_NB_WORDS, EMBEDDING_DIM)

allaccuracylist=[]
logger.info('Set TF configuration')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
config.log_device_placement=False
set_session(tf.Session(config=config))

for r in range(0,args.timesteps):
    accuracylist=[]
    logger.info(" * Validation fold: {}".format(str(r)))
    logger.info('Repetition:'+str(r+1))

    x_la, y_la, x_un, y_un = utils.partition_data(data, labels, args.label_data_size, shuffle=True)
    x_trn, y_trn, x_valtest, y_valtest = utils.partition_data(x_la, y_la, args.initial_training_size ,
                                                              shuffle=True)
    x_val, y_val, x_test, y_test = utils.partition_data(x_valtest, y_valtest, args.validation_size,
                                                        shuffle=True)
    x_pool = list(x_un)
    y_pool = list(y_un)
    logger.info(
        "[Repition {}] Partition data: labeled = {}, val = {}, test = {}, unlabeled pool = {} ".format(str(r),
                                                                                                       len(x_trn),
                                                                                                       len(x_val),
                                                                                                       len(x_test),
                                                                                                       len(x_pool)))

    classifer = getClassifier(num_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH)
    querydata=[]
    querylabels=[]
    if args.initial_training_size > 0:
        classifer.fit(x_trn, y_trn, validation_data=(x_val, y_val),
                  batch_size=args.classifier_batch_size, epochs=args.classifier_epochs, verbose=0)
        accuracy = classifer.evaluate(x_test, y_test, verbose=0)[2]
        accuracylist.append(accuracy)
        logger.info(' * Labeled data size: {}'.format(str(len(x_trn))))
        logger.info(" [Step 0] Accurary : {}".format(str(accuracy)))
    
    querydata = querydata + list(x_trn)
    querylabels = querylabels + list(y_trn)
    logger.info('Model initialized...')

    for t in range(0,BUDGET):
        logger.info('Repetition:'+str(r+1)+' Iteration '+str(t+1))
        logger.info('Number of current samples:'+str((t+1)*numofsamples))
        sampledata=[]
        samplelabels=[]
        if(QUERY == 'Random'):
            sampledata, samplelabels, x_pool, y_pool= randomSample(x_pool, y_pool, numofsamples)
        elif(QUERY == 'Uncertainty'):
            sampledata, samplelabels, x_pool, y_pool= uncertaintySample(x_pool, y_pool, numofsamples, classifer)
        elif(QUERY == 'Diversity'):
            sampledata, samplelabels, x_pool, y_pool= diversitySample(x_pool, y_pool, numofsamples, querydata)
        querydata=querydata+sampledata
        querylabels=querylabels+samplelabels
        
        x_train = np.array(querydata)
        y_train = np.array(querylabels)
         
        classifer.fit(x_train, y_train, validation_data=(x_val, y_val),
                      batch_size=args.classifier_batch_size, epochs=args.classifier_epochs, verbose=0)

        if((t+1) % 5 == 0):
            accuracy = classifer.evaluate(x_test, y_test, verbose=0)[2]
            accuracylist.append(accuracy)
            logger.info('[Learning phase] Budget used so far: {}'.format(str(t)))
            logger.info(' * Labeled data size: {}'.format(str(len(x_train))))
            logger.info(" [Step {}] Accurary : {}".format(str(t), str(accuracy)))
    allaccuracylist.append(accuracylist)
    classifiername = "{}/{}_classifier_fold_{}.h5".format(args.output, DATASET_NAME, str(r))
    classifer.save(classifiername)
    logger.info(" * End of fold {}. Clear session".format(str(r)))
    K.clear_session()
    del classifer
    gc.collect()

    accuracyarray=np.array(allaccuracylist)
    averageacc=list(np.mean(accuracyarray, axis=0))
    logger.info('Accuray list: ')
    logger.info(averageacc)
    ww=open(resultname,'w')
    ww.writelines(str(line)+ "\n" for line in averageacc)
    ww.close()

logger.info(resultname)
