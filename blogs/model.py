import tensorflow as tf

'''
C = tf.placeholder(dtype = tf.float32, shape = [None, 500, 50])
W = tf.placeholder(dtype = tf.float32, shape = [None, 300, 300,1])
T = tf.placeholder(dtype = tf.float32, shape = [None, 200])
'''

def model(INPUT,SIZE,DROP):
    max_char = 10
    max_word = 200
    n_dimension = 300    
    n_chars = 50
    n_topics = 200
    n_author = 19320
    n_gender = 2
    n_age = 3
    n_job = 40
    n_lstm_hidden = 128
    n_conv_channels = 256
#Variable of topic network
    w_topic = tf.Variable(
            tf.truncated_normal(
                [n_topics, 256],
                stddev = 0.1
                )
            )
    b_topic = tf.Variable(
            tf.constant(
                1 / (256),
                shape = [256]
                )
            )

#Variable of char network
    lstmcell = tf.contrib.rnn.LSTMCell(
        n_lstm_hidden,
        state_is_tuple = True
        )
    lstmcell2 = tf.contrib.rnn.LSTMCell(
        n_lstm_hidden,
        state_is_tuple = True
        )
    w_char = tf.Variable(
            tf.truncated_normal(
                [n_lstm_hidden, 256],
                stddev = 0.1
                )
            )
    b_char = tf.Variable(
            tf.constant(
                1 / (256),
                shape = [256]
                )
            )

#Variable of word network
    w_conv = tf.Variable(
            tf.truncated_normal(
                shape = [3, n_dimension, 1, n_conv_channels],
                stddev = 0.1
                )
            )

    w_word = tf.Variable(
            tf.truncated_normal(
                shape = [n_conv_channels, 256],
                stddev = 0.1
                )
            )
    b_word = tf.Variable(
            tf.constant(
                1 / (256),
                shape = [256]
                )
            )

#Variable of classifier
    w_fc = tf.Variable(
            tf.truncated_normal(
                shape = [256, 4 * 512],
                stddev = 0.1
                )
            )
    b_fc = tf.Variable(
            tf.constant(
                1 / (4 * 512),
                shape = [4 * 512]
                )
            )

    w_classifier_author = tf.Variable(
            tf.truncated_normal(
                shape = [512, n_author],
                stddev = 0.1
                )
            )
    b_classifier_author = tf.Variable(
            tf.constant(
                1 / n_author,
                shape = [n_author]
                )
            )

    w_classifier_gender = tf.Variable(
            tf.truncated_normal(
                shape = [512, n_gender],
                stddev = 0.1
                )
            )
    b_classifier_gender = tf.Variable(
            tf.constant(
                1 / n_gender,
                shape = [n_gender]
                )
            )
    w_classifier_age = tf.Variable(
            tf.truncated_normal(
                shape = [512, n_age],
                stddev = 0.1
                )
            )
    b_classifier_age = tf.Variable(
            tf.constant(
                1 / n_age,
                shape = [n_age]
                )
            )

    w_classifier_job = tf.Variable(
            tf.truncated_normal(
                shape = [512, n_job],
                stddev = 0.1
                )
            )
    b_classifier_job = tf.Variable(
            tf.constant(
                1 / n_job,
                shape = [n_job]
                )
            )

    C, W, T = INPUT
    KEEP_CHAR,KEEP_WORD, KEEP_TOPIC, KEEP_CLASSIFIER = DROP
    def char_model(chars = C):
        print("\tChar Model Build Start.")
        out = []
        print("\t\tLstm Model Build Start.")
        with tf.variable_scope("CharLstm"):
            for i in range(max_word):
                print("\t\t\tword_step in",i)
                state = lstmcell.zero_state(SIZE,dtype = tf.float32)
                for j in range(max_char):
                    print("\t\t\t\tchar_step in",j)
                    if i != 0 or j != 0: tf.get_variable_scope().reuse_variables()
                    out_, state = lstmcell(chars[:,i,j,:],state)
                _,state = state
                out.append(tf.reshape(state,shape = [-1,1,n_lstm_hidden]))
        out = tf.nn.dropout(tf.concat(out,axis = 1),KEEP_CHAR)
        with tf.variable_scope("WordLstm"):
            for i in range(max_word):
                state = lstmcell2.zero_state(SIZE,dtype = tf.float32)
                if i != 0:tf.get_variable_scope().reuse_variables()
                out_,state = lstmcell2(out[:,i,:],state)
        _,state = state
        out = tf.reshape(state,shape = [-1,n_lstm_hidden])
        print(out.shape)
        print("\t\tLstm Model Done.")

#        print("\t\tPool Build Start.")
#        out = tf.nn.max_pool(
#                out,
#                ksize = [1,max_word,1,1],
#                strides = [1,1,1,1],
#                padding = 'VALID'
#                )
#        out = tf.reshape(
#                out,
#                [-1,n_lstm_hidden]
#                )
#        print("\t\tPool Done.")

        print("\t\tFC Build Start.")
        out = tf.tanh(
            tf.matmul(out,w_char
                ) + b_char
            )
        print("\t\tFC Done.")
        print("\tChar Model Done.")
        return out 
    
    def word_model(words = W):
        print("\tWord Model Build Start.")
        print("\t\tCONV Building Start.")
        out = tf.nn.conv2d(
                words,
                w_conv,
                strides = [1,1,n_dimension,1],
                padding = 'SAME'
                )
        out = tf.nn.dropout(
                out,
                KEEP_WORD
                )
        print("\t\tCONV Done")
        print("\t\tPOOL Build Start.")
        out = tf.nn.max_pool(
                out,
                ksize = [1,max_word,1,1],
                strides = [1,1,1,1],
                padding = 'VALID'
                )
        print("\t\tPOOL Done.")
        '''
        state = w_lstm.zero_state(SIZE,dtype = tf.float32)
        out = None
        word = tf.reshape(words,shape = [-1,max_word,n_dimension])
        with tf.variable_scope("lstm_"):
            for step in range(max_word):
                print("\t\t\tStep in",step)
                if step > 0: tf.get_variable_scope().reuse_variables()
                out, state = w_lstm(word[:, step, :], state) 
        '''
        print("\t\tWord FC Build Start.")
        out = tf.reshape(
                out,
                [-1,n_conv_channels]
                )
        out = tf.tanh(
            tf.matmul(out,w_word
                ) + b_word
            )
        print("\t\tWord FC Done.")
        print("\tWord Model Done.")
        return out

    def topic_model(topics = T):
        print("\tTopic Model Build Start.")
        print("\t\tTopic FC Build Start.")
        topics = tf.nn.dropout(
                topics,
                KEEP_TOPIC
                )
        out = tf.tanh(
            tf.matmul(topics,w_topic
                ) + b_topic
            )
        print("\t\tTopic FC Done.")
        print("\tTopic Model Done.")
        return out 

    def classifier(chars,words,topics):
        print("\tClassifier Build Start.")
        def cat(f_char,f_word,f_topic):
            #f_char = tf.reshape(f_char,shape = [-1,n_multiple * n_author,1,1])
            #f_word = tf.reshape(f_word,shape = [-1,n_multiple * n_author,1,1])
            #f_topic = tf.reshape(f_topic,shape = [-1,n_multiple * n_author,1,1])
            #return tf.concat([f_char,f_word,f_topic],axis = 2)
            return f_word * f_char * f_topic 
        print("\t\tConcat Start.")
        input_ = cat(
                chars,
                words,
                topics
                )
        input_ = tf.nn.dropout(
                input_,
                KEEP_CLASSIFIER
                )
        print("\t\tConcat Done.")
        print("\t\tClassifier FC Start.")

        fc_out = tf.tanh(
                tf.matmul(
                    input_,
                    w_fc
                    ) + b_fc
                )
        print("\t\tClassifier FC Done.")
        print("\tClassifier Done.")
        fc_out = tf.nn.dropout(
                fc_out,
                KEEP_CLASSIFIER
                )
        out_author = tf.nn.relu(
                tf.matmul(
                    fc_out[:,:512],
                    w_classifier_author
                    ) + b_classifier_author
                )
        out_gender = tf.nn.relu(
                tf.matmul(
                    fc_out[:,512:1024],
                    w_classifier_gender
                    ) + b_classifier_gender
                )
        out_age = tf.nn.relu(
                tf.matmul(
                    fc_out[:,1024:1536],
                    w_classifier_age
                    ) + b_classifier_age
                )
        out_job = tf.nn.relu(
                tf.matmul(
                    fc_out[:,1536:],
                    w_classifier_job
                    ) + b_classifier_job
                )
        out = out_author,out_gender,out_age,out_job    
        return out
    print("Graph Build Start.")
    res = classifier(
            char_model(C),
            word_model(W),
            topic_model(T)
            )
    print("Graph Done.")
    return res
