# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:03:05 2017

@author: lming
"""
import gc
import time

import utils
from model import *
from keras.models import Sequential, Model, load_model
from queryStrategy import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

start_time = time.time()
args = utils.get_args()
logger = utils.init_logger()

EMBEDDING_DIM = args.embedding_dim
MAX_SEQUENCE_LENGTH = args.max_seq_length
MAX_NB_WORDS = args.max_nb_words

rootdir=args.root_dir
DATASET_NAME=args.dataset_name
TEXT_DATA_DIR = args.text_data_dir
TEST_DIR = args.test_set
GLOVE_DIR = args.word_vec_dir

QUERY=args.query_strategy
EPISODES=args.episodes
k_num=args.k
BUDGET=args.annotation_budget

policyname=args.policy_path
resultname= "{}/{}_accuracy.txt".format(args.output, DATASET_NAME)
if not policyname:
    raise Exception("Missing pretrained AL policy path")

logger.info("Transfer AL policy [{}] to task on dataset {}".format(QUERY, DATASET_NAME))
logger.info(" * POLICY path: {}".format(policyname))
logger.info(" * INPUT directory: {}".format(TEXT_DATA_DIR))
logger.info(" * OUTPUT file: {}".format(resultname))

# first, build index mapping words in the embeddings set
# to their embedding vector
embeddings_index = utils.load_embeddings(GLOVE_DIR)

# second, prepare text samples and their labels
data, labels, word_index = utils.load_data(TEXT_DATA_DIR, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
# test_data, test_labels, _ = utils.load_data(TEST_DIR, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
# test_data, test_labels = utils.shuffle_test_data(test_data, test_labels)
#data set for inisialize the model

embedding_matrix, num_words = utils.construct_embedding_table(embeddings_index, word_index, MAX_NB_WORDS, EMBEDDING_DIM)

logger.info('Set TF configuration')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
config.log_device_placement=False
set_session(tf.Session(config=config))


logger.info('Begin transfering policy..')
allaccuracylist=[]  
for tau in range(0, args.timesteps):
    logger.info(" * Validation times: {}".format(str(tau)))
    logger.info("[Repition {}] Load policy from {}".format(str(tau), policyname))
    policy = load_model(policyname)

    accuracylist=[]
    #Shuffle D_L
    x_la, y_la, x_un, y_un = utils.partition_data(data, labels, args.label_data_size, shuffle=True)
    x_trn, y_trn, x_valtest, y_valtest = utils.partition_data(x_la, y_la, args.initial_training_size,
                                                              shuffle=True)
    x_val, y_val, x_test, y_test = utils.partition_data(x_valtest, y_valtest, args.validation_size,
                                                        shuffle=True)
    x_pool = list(x_un)
    y_pool = list(y_un)
    logger.info(
        "[Repition {}] Partition data: labeled = {}, val = {}, test = {}, unlabeled pool = {} ".format(str(tau),
                                                                                                       len(x_trn),
                                                                                                       len(x_val),
                                                                                                       len(x_test),
                                                                                                       len(x_pool)))
    #Initilize classifier
    model= getClassifier(num_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH,
                         model_path=args.model_path, learning_rate=args.classifier_learning_rate)
    if args.initial_training_size > 0:
        model.fit(x_trn, y_trn, validation_data=(x_val, y_val),
              batch_size=args.classifier_batch_size, epochs=args.classifier_epochs)
        accuracy = model.evaluate(x_test, y_test, verbose=0)[2]
        accuracylist.append(accuracy)
        logger.info(' * Labeled data size: {}'.format(str(len(x_trn))))
        logger.info(" [Step 0] Accurary : {}".format(str(accuracy)))

    #In every episode, run the trajectory
    for t in range(0,BUDGET):
        logger.info('Episode:'+str(tau+1)+' Budget:'+str(t+1))
        x_new=[]
        y_new=[]
        
        loss=10
        row=0
        bestindex=0
        '''
        queryscores=[]
        for i in range(0, (len(x_pool)/k_num)):
            temp_x_pool=x_pool[(k_num*i):(k_num*i+k_num)]
            temp_y_pool=y_pool[(k_num*i):(k_num*i+k_num)]
            state=getAState(x_trn, y_trn,temp_x_pool,model)
            tempstates= np.expand_dims(state, axis=0)
            tempscores=get_intermediatelayer(policy, 5, tempstates)[0]
            queryscores=queryscores+tempscores
            print(tempscores)
        '''
        #Random sample k points from D_un
        x_rand_unl, y_rand_unl, queryindices = randomKSamples(x_pool, y_pool, k_num)
        
        #Use the policy to get best sample
        state=getAState(x_trn, y_trn,x_rand_unl,model)
        tempstates= np.expand_dims(state, axis=0)
        a=policy.predict(tempstates)
        action=policy.predict_classes(tempstates, verbose=0)[0]
        x_new=x_rand_unl[action]
        y_new=y_rand_unl[action]
        
        x_trn=np.vstack([x_trn,x_new])
        y_trn=np.vstack([y_trn,y_new])
        model.fit(x_trn, y_trn, validation_data=(x_val, y_val),
                  batch_size=args.classifier_batch_size, epochs=args.classifier_epochs)
        
        index_new=queryindices[action]
        del x_pool[index_new]
        del y_pool[index_new]

        if((t+1) % 5 == 0):
            accuracy=model.evaluate(x_test, y_test,verbose=0)[2]
            accuracylist.append(accuracy)
            logger.info('[Learning phase] Budget used so far: {}'.format(str(t)))
            logger.info(' * Labeled data size: {}'.format(str(len(x_trn))))
            logger.info(" [Step {}] Accurary : {}".format(str(t), str(accuracy)))
    allaccuracylist.append(accuracylist)

    classifiername = "{}/{}_classifier_fold_{}.h5".format(args.output, DATASET_NAME, str(tau))
    model.save(classifiername)
    logger.info(" * End of fold {}. Clear session".format(str(tau)))
    K.clear_session()
    del model
    gc.collect()

    accuracyarray=np.array(allaccuracylist)
    averageacc=list(np.mean(accuracyarray, axis=0))
    ww=open(resultname,'w')
    ww.writelines(str(line)+ "\n" for line in averageacc)
    ww.close()
    logger.info("Transfer complete")