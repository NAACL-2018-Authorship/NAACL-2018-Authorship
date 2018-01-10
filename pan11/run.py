import tensorflow as tf
from sklearn.externals import joblib
import numpy as np
import sklearn.metrics as sk
from conf import conf
from model import model
from preprocess import *

data_path = "../list/"
dic_path = "../dic/"
model_path = "../model/"
filename = "Large"
trainfile = data_path + filename + "Train.json"
testfile = data_path + filename + "Test.json"
#ldamodelfile = model_path + "lda/LDA_model_" + filename + "Train_S_" + str(150) + ".m"
ldamodelfile = model_path + "lda/LDA_model_LargeTrain_S_150_ac40.6.m"
authordicfile = dic_path + filename + "Train_author.json"
chardicfile = dic_path + "char_dic.json"
worddicfile = dic_path + "word_dic_" + filename + ".json"
wordembeddingdicfile = dic_path + "word_embedding_dic.json"

train_data = get_data(trainfile)
test_data = get_data(testfile)
lda_model = joblib.load(ldamodelfile)
author_dic = get_data(authordicfile)
char_dic = get_data(chardicfile)
word_dic = get_data(worddicfile)
word_embedding_dic = get_data(wordembeddingdicfile)

TRAIN = Data(train_data, lda_model, author_dic, char_dic, word_dic, word_embedding_dic)
TEST = Data(test_data, lda_model, author_dic, char_dic, word_dic, word_embedding_dic)

def ACC(logits,labels):
    res = tf.argmax(logits,axis = 1)
    target = tf.argmax(labels,axis = 1)
    acc  = tf.reduce_mean(tf.cast(tf.equal(res, target),tf.float32))
    return acc

last_epoch = tf.Variable(0,dtype=tf.int32,trainable=False)

def train(max_epoch = 100, force = False, ID = 1):
    LR = conf[ID].lr
    lr = tf.Variable(LR,dtype = tf.float32,trainable=False)
    n_batch = conf[ID].n_batch
    max_char = conf[ID].max_char
    max_word = conf[ID].max_word
    n_dimension = conf[ID].n_dimension
    n_chars = conf[ID].n_chars
    n_topics = conf[ID].n_topics
    n_author = conf[ID].n_author
    n_lstm_hidden = conf[ID].n_lstm_hidden
    n_conv_channels = conf[ID].n_conv_channels
    keep_char = conf[ID].keep_char
    keep_word = conf[ID].keep_word
    keep_topic = conf[ID].keep_topic
    keep_classifier = conf[ID].keep_classifier

    C = tf.placeholder(dtype = tf.float32, shape = [None, max_word, max_char, n_chars])
    W = tf.placeholder(dtype = tf.float32, shape = [None, max_word, n_dimension,1])
    T = tf.placeholder(dtype = tf.float32, shape = [None, n_topics]) 
    L = tf.placeholder(dtype = tf.int32, shape = [None, n_author])
    KC = tf.placeholder(dtype = tf.float32)
    KW = tf.placeholder(dtype = tf.float32)
    KT = tf.placeholder(dtype = tf.float32)
    KCL = tf.placeholder(dtype = tf.float32)
    S = tf.placeholder(dtype = tf.int32)

    INPUT = (C,W,T)
    KEEP_PROB = (KC,KW,KT,KCL)
    SIZE = S
    LABEL = L

    path = model_path + "full_" + str(ID) + "/"

    f = open(path + "train.log","w")
    t = open("test.log","a")
    gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction = 0.85
            )
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session() as sess:
        logit = model(INPUT = INPUT, KEEP_PROB = KEEP_PROB, SIZE = SIZE, ID = ID)
        loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits = logit,
                    labels = LABEL
                    )
                )
        acc = ACC(
                logit,
                LABEL
                )
        print("Optimizer Build Start.")
        with tf.variable_scope(tf.get_variable_scope(),reuse=False):
            optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
        print("Optimizer Done.")
        print("Saver Build Start.")
        saver = tf.train.Saver(tf.global_variables())
        print("Saver Done.")
        print("Initialize Start.")
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path and not force:
            saver.restore(sess, ckpt.model_checkpoint_path)
            #saver.restore(sess, path+"model-1400")
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
        print("Initialize Done.")
        for epoch in range(sess.run(last_epoch) + 1, sess.run(last_epoch) + max_epoch + 1):
            sess.run(tf.assign(lr,  tf.minimum(lr ** 0.995,0.0008)))
            batch = 0
            end = False
            while not end:
                chars,words,topics,labels,end = TRAIN.next_batch(n_batch,max_char,max_word)
                train_loss, train_acc, _ =  sess.run(
                        [loss,acc,optimizer],
                        feed_dict = {C:chars, W:words, T:topics, L:labels, KC:keep_char, KW:keep_word, KT:keep_topic, KCL:keep_classifier, S:n_batch}
                        )
                print(epoch,batch,":",train_loss,train_acc)
                f.write(str(epoch)+ ' ' + str(batch)+':'+"loss: " + str(train_loss)+ " acc: " + str(train_acc) + "\n")
                batch += 1
            if epoch % 10 == 0:
                sess.run(tf.assign(last_epoch,epoch))
                saver.save(sess, path + "model", global_step = epoch)
            chars,words,topics,labels,_ = TEST.next_batch(TEST.cnt,max_char,max_word)
            test_loss,test_acc,y_pred = sess.run(
                    [loss,acc,tf.argmax(logit,1)],
                    feed_dict = {C:chars, W:words, T:topics, L:labels, KC:1., KW:1., KT:1., KCL:1., S:TEST.cnt}
                    )
            y_true = np.argmax(labels,1)
            precision = sk.precision_score(y_true,y_pred,average="macro")
            recall = sk.recall_score(y_true,y_pred,average="macro")
            f1 = sk.f1_score(y_true,y_pred,average="macro")
            print("#test: MODEL:"+str(ID)+" EPOCH:" + str(epoch) + " LOSS:" + str(test_loss) + " ACC:" + str(test_acc) +" "+ str(precision)+" " + str(recall) +" "+ str(f1)+"\n")
            t.write("#test: MODEL:"+str(ID)+" EPOCH:" + str(epoch) + " LOSS:" + str(test_loss) + " ACC:" + str(test_acc)+" "+ str(precision)+" " + str(recall) + " " + str(f1)+ "\n")
    f.close()
    t.close()
train(max_epoch = 300,force = True,ID=31)
#train(max_epoch = 50,force = False,ID=7)
#test(ID = 1)
