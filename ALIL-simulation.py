# -*- coding: utf-8 -*-
"""
Created on Thu Nov 09 15:05:23 2017

@author: lming
"""
import gc

from keras.models import load_model

from model import *
from keras.utils import to_categorical
import time
from queryStrategy import *
import utils
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
GLOVE_DIR = args.word_vec_dir

QUERY=args.query_strategy
EPISODES=args.episodes
timesteps=args.timesteps
k_num=args.k
BUDGET=args.annotation_budget

policyname="{}/{}_policy.h5".format(args.output, DATASET_NAME)
classifiername="{}/{}_classifier.h5".format(args.output, DATASET_NAME)


# first, build index mapping words in the embeddings set
# to their embedding vector
embeddings_index = utils.load_embeddings(GLOVE_DIR)

# second, prepare text samples and their labels
data, labels, word_index = utils.load_data(TEXT_DATA_DIR, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)

# prepare embedding matrix
embedding_matrix, num_words = utils.construct_embedding_table(embeddings_index, word_index, MAX_NB_WORDS, EMBEDDING_DIM)

logger.info('Set TF configuration')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
config.log_device_placement=False
set_session(tf.Session(config=config))

logger.info('Begin training active learning policy..')
#load random initialised policy
policy=getPolicy(k_num,104)
policy.save(policyname)
#Memory (two lists) to store states and actions
states=[]
actions=[]
for tau in range(0,EPISODES):
    #partition data
    logger.info(" * Start episode {}".format(str(tau)))
    logger.info("[Ep {}] Split data to train, validation and unlabeled")
    x_la, y_la, x_un, y_un = utils.partition_data(data, labels, args.label_data_size, shuffle=True)

    #Split initial train,  validation set
    x_trn, y_trn, x_val, y_val = utils.partition_data(x_la, y_la, args.initial_training_size,
                                                              shuffle=True)
    
    x_pool=list(x_un)
    y_pool=list(y_un)

    logger.info("[Episode {}] Load Policy from path {}".format(str(tau), policyname))
    policy = load_model(policyname)

    #Initilize classifier
    model= getClassifier(num_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH)
    initial_weights = model.get_weights()
    if args.initial_training_size > 0:
        model.fit(x_trn, y_trn, batch_size=args.classifier_batch_size, epochs=args.classifier_epochs, verbose=0)
    current_weights = model.get_weights()
    model.save(classifiername)

    #toss a coint
    coin=np.random.rand(1)
    #In every episode, run the trajectory
    for t in range(0,BUDGET):
        if t % 10 == 0:
            logger.info('Episode:'+str(tau+1)+' Budget:'+str(t+1))
        x_new=[]
        y_new=[]
        accuracy = -1
        row=0
        #save the index of best data point or acturally the index of action
        bestindex=0 
        #Random sample k points from D_pool
        x_rand_unl, y_rand_unl, queryindices = randomKSamples(x_pool, y_pool, k_num)
        if len(x_rand_unl) == 0:
            logger.info(" *** WARNING: Empty samples")
        for datapoint in zip(x_rand_unl, y_rand_unl):
            model.set_weights(initial_weights)
            model.fit(x_trn, y_trn,batch_size=args.classifier_batch_size, epochs=args.classifier_epochs, verbose=0)
            x_temp=datapoint[0]
            y_temp=datapoint[1]
            x_temp_trn= np.expand_dims(x_temp, axis=0)
            y_temp_trn=np.expand_dims(y_temp, axis=0)
            
            history= model.fit(x_temp_trn, y_temp_trn, validation_data=(x_val, y_val),
                               batch_size=args.classifier_batch_size, epochs=args.classifier_epochs, verbose=0)
            val_accuracy = history.history['val_acc'][0]
            if(val_accuracy>accuracy):
                bestindex=row
                accuracy=val_accuracy
                x_new=x_temp
                y_new=y_temp
            row=row+1
        model.set_weights(current_weights)
        state=getAState(x_trn, y_trn,x_rand_unl,model)

        # if head(>0.5), use the policy; else tail(<=0.5), use the expert
        if(coin>0.5):
            logger.debug(' * Use the POLICY [coin = {}]'.format(str(coin)))
            #tempstates= np.ndarray((1,K,len(state[0])), buffer=np.array(state))
            tempstates= np.expand_dims(state, axis=0)
            action=policy.predict_classes(tempstates)[0]
        else:
            logger.debug(' * Use the EXPERT [coin = {}]'.format(str(coin)))
            action=bestindex
        states.append(state)
        actions.append(action)
        x_trn=np.vstack([x_trn, x_rand_unl[action]])
        y_trn=np.vstack([y_trn, y_rand_unl[action]])
        model.fit(x_trn, y_trn, batch_size=args.classifier_batch_size, epochs=args.classifier_epochs, verbose=0)
        current_weights = model.get_weights()
        model.save(classifiername)

        index_new=queryindices[action]
        del x_pool[index_new]
        del y_pool[index_new]
        
    cur_states=np.array(states)
    cur_actions=to_categorical(np.asarray(actions), num_classes=k_num)
    train_his = policy.fit(cur_states, cur_actions)
    logger.info(" [Episode {}] Training policy loss = {}, acc = {}, mean_squared_error = {}".
                format(tau, train_his.history['loss'][0], train_his.history['acc'][0],
                       train_his.history['mean_squared_error'][0]))
    logger.info(" * End episode {}. Save policy to {}".format(str(tau), policyname))
    policy.save(policyname)
    K.clear_session()
    del model
    del x_trn
    del y_trn
    del x_val
    del y_val
    del x_pool
    del y_pool
    del initial_weights
    del current_weights
    gc.collect()

print("--- %s seconds ---" % (time.time() - start_time))
logger.info("ALIL simulation completed")
del policy

