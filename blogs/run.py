import numpy as np
import tensorflow as tf
from lda import LDA
from sklearn.externals import joblib
import json
import gc

from preprocess import data
from model import model

ldamodel = joblib.load("../model/blog_lda_200.m")
path = "../model/3/"
def ACC(logits,labels):
    res = tf.argmax(logits,axis = 1)
    target = tf.argmax(labels,axis = 1)
    acc  = tf.reduce_mean(tf.cast(tf.equal(res, target),tf.float32))
    return acc


def train(max_epoch = 100, force = False, ID = 1):
    with open('../data_new/a2t.json','r') as f:
        a2t = json.load(f)
    with open('../data_new/it2e.json','r') as f:
        it2e = json.load(f)
    with open('../data_new/it2c.json','r') as f:
        it2c = json.load(f)
    dics = (it2e,it2c,a2t)
    n_batch = 48
    LR = 0.008

    lr = tf.Variable(LR,dtype = tf.float32,trainable = False)
    last_epoch = tf.Variable(0,dtype = tf.int32,trainable = False)
    C = tf.placeholder(dtype = tf.float32, shape = [None, 200, 10, 50])
    W = tf.placeholder(dtype = tf.float32, shape = [None, 200, 300])
    w = tf.reshape(W,[-1,200,300,1])
    T = tf.placeholder(dtype = tf.float32, shape = [None, 200])
    AL = tf.placeholder(dtype = tf.int32, shape = [None, 19320])
    PL = tf.placeholder(dtype = tf.int32, shape = [None, 2 + 3 + 40])
    KC = tf.placeholder(dtype = tf.float32)
    KW = tf.placeholder(dtype = tf.float32)
    KT = tf.placeholder(dtype = tf.float32)
    KCL = tf.placeholder(dtype = tf.float32)

    D = (KC,KW,KT,KCL)
    INPUT = (C,w,T)

    SIZE = tf.placeholder(dtype = tf.int32)

#    gpu_options = tf.GPUOptions(
#                per_process_gpu_memory_fraction = True
#            )
# config = tf.ConfigProto(gpu_options=gpu_options)
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        with tf.device("/gpu:0"): 
            logit_author,logit_gender,logit_age,logit_job= model(INPUT = INPUT, SIZE = SIZE,DROP = D)
            print("model done")
#loss0 = 10 * tf.reduce_mean((TL ** (1/2) - topic_logit ** (1/2)) ** 2)
            loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits = logit_author,
                    labels = AL
                    ))
            loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits = logit_gender,
                    labels = PL[:,:2]
                    )) + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits = logit_age,
                    labels = PL[:,2:2+3]
                    )) + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits = logit_job,
                    labels = PL[:,2+3:2+3+40]
                    ))


        with tf.device("/gpu:0"):
            optimizer1 = tf.train.AdamOptimizer(lr).minimize(loss1+loss2)


        acc1 = ACC(logit_author,AL)



        tf.summary.scalar('loss1',loss1)
        tf.summary.scalar('loss2',loss2)
        tf.summary.scalar('gender_acc',acc1)

        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter('../result/train')

        print("summary done")
        saver = tf.train.Saver(tf.global_variables())
        print("saver done")
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path and not force:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore")
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
            print("init")
        steps = 0
        a = 0
        b = 1
        c = 1
        for epoch in range(sess.run(last_epoch) + 1, sess.run(last_epoch) + max_epoch + 1):
            sess.run(tf.assign(lr, tf.maximum(lr ** 0.98,0.001)))
            batch = 0            
            for i in range(35):
                print("in",epoch,batch)
                print("loda data")
                TRAIN = data("../data_new/train/" + str(i) + ".json","../d2t_new/train/" + str(i) + ".json",dics)
                print("done")
                print(TRAIN.size)
                end = False
                while not end:
                    print("geting batch")
                    end,batch_size,chars,words,topics,author_label,profile_label,topic_label = TRAIN.next_batch(n_batch,200,10)
                    print("done\nrun")
#train_loss1,train_loss2,train_loss3,train_acc1,train_acc2,train_acc3,train_acc4,train_acc5,_,__,___,summary = sess.run(
#                            [loss1,loss2,loss3,acc1,acc2,acc3,acc4,acc5,optimizer1,optimizer2,optimizer3,merged],
#                            feed_dict = {C:chars,W:words,T:topics,AL:author_label,PL:profile_label, S:batch_size}
#                        )
                    print(batch_size)
                    train_loss1,train_loss2,train_acc1,_,summary= sess.run(                        

                            [loss1,loss2,acc1,optimizer1,merged],
                            feed_dict = {C:chars,W:words,T:topics,AL:author_label,PL:profile_label, SIZE:batch_size,KC:0.4,KW:0.4,KT:0.4,KCL:0.4}
                        )
                    
                    print("done")

                    print("epoch",epoch,"batch",batch,":")
                    print("loss:",train_loss1,train_loss2)
                    print("acc",train_acc1)
#print("author_acc:",train_acc4)
#print("after_profile:",train_acc4,"not",train_acc5)
                    train_writer.add_summary(summary,steps)
                    batch += 1
                    steps += 1
                
                
                del TRAIN, end,batch_size,chars,words,topics,author_label,profile_label
                gc.collect()
            sess.run(tf.assign(last_epoch,epoch))
            saver.save(sess, path + "model", global_step = epoch)
            if epoch % 1 == 0:
                print("validing...")
                closs1 = 0.
                closs2 = 0.
                closs3 = 0.
                cacc1 = 0.
                cacc2 = 0.
                cacc3 = 0.
                cacc4 = 0.
                cacc5 = 0.
                cnt = 0.
                for i in range(5):
                    print("in " + str(i))

                    TEST =  data("../data_new/valid/" + str(i) + ".json","../d2t_new/valid/" + str(i) + ".json",dics)
                    cnt += TEST.size
                    end = False
                    while not end:
                        end,batch_size,chars,words,topics,author_label,profile_label,_ = TEST.next_batch(TEST.size,200,10)
#                    test_loss1,test_loss2,test_loss3,test_acc1,test_acc2,test_acc3,test_acc4,test_acc5,_,__,___,summary = sess.run(
##                        [loss1,loss2,loss3,acc1,acc2,acc3,acc4,acc5,optimizer1,optimizer2,optimizer3,merged],
#                        feed_dict = {C:chars,W:words,T:topics,AL:author_label,PL:profile_label,S:batch_size}
#                        )
                        test_loss1,test_loss2,test_acc1 = sess.run(                        
                            [loss1,loss2,acc1],
                            feed_dict = {C:chars,W:words,T:topics,AL:author_label,PL:profile_label, SIZE:batch_size,KC:1.,KW:1.,KT:1.,KCL:1.}
                        )
                        closs1 += test_loss1 * batch_size
                        closs2 += test_loss2 * batch_size
                        cacc1 += test_acc1 * batch_size
                    del TEST,end,batch_size,chars,words,topics,author_label,profile_label

                    gc.collect()
                test_loss1 = closs1 / cnt
                test_loss2 = closs2 / cnt
                test_acc1 = cacc1 / cnt
                print("valid_epoch",epoch,":")
                print("loss:",test_loss1,test_loss2)
                print("acc",test_acc1)
#print("author_acc:",test_acc4)
                with open('valid.res','a') as f:
                    f.write(str(ID) + " " +  str(test_loss1) +" "+ str(test_acc1) + "\n")


            if epoch % 1 == 0:
                print("testing...")
                closs1 = 0.
                closs2 = 0.
                closs3 = 0.
                cacc1 = 0.
                cacc2 = 0.
                cacc3 = 0.
                cacc4 = 0.
                cacc5 = 0.
                cnt = 0.
                for i in range(10):
                    print("in " + str(i))
                    TEST =  data("../data_new/test/" + str(i) + ".json","../d2t_new/test/" + str(i) + ".json",dics)
                    cnt += TEST.size
                    end = False
                    while not end:
                        end,batch_size,chars,words,topics,author_label,profile_label,_ = TEST.next_batch(TEST.size,200,10)
#                    test_loss1,test_loss2,test_loss3,test_acc1,test_acc2,test_acc3,test_acc4,test_acc5,_,__,___,summary = sess.run(
##                        [loss1,loss2,loss3,acc1,acc2,acc3,acc4,acc5,optimizer1,optimizer2,optimizer3,merged],
#                        feed_dict = {C:chars,W:words,T:topics,AL:author_label,PL:profile_label,S:batch_size}
#                        )
                        test_loss1, test_loss2,test_acc1= sess.run(                        
                            [loss1,loss2,acc1],
                            feed_dict = {C:chars,W:words,T:topics,AL:author_label,PL:profile_label, SIZE:batch_size,KC:1.,KW:1.,KT:1.,KCL:1.}
                        )
                        closs1 += test_loss1 * batch_size
                        closs2 += test_loss2 * batch_size
                        cacc1 += test_acc1 * batch_size
                    del TEST,end,batch_size,chars,words,topics,author_label,profile_label

                    gc.collect()
                test_loss1 = closs1 / cnt
                test_loss2 = closs2 / cnt

                test_acc1 = cacc1 / cnt

                print("test_epoch",epoch,":")
                print("loss:",test_loss1,test_loss2)
                print("acc:",test_acc1)

                with open('test.res','a') as f:
                    f.write(str(ID) + " " +  str(test_loss1) +" "+ str(test_acc1)+ "\n")


train(20,True,3)
