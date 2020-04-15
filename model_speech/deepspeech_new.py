# this model is baidu deepspeech2 ASR model

import keras
from keras.layers import Reshape, Lambda, Dropout
from keras.layers.merge import add, concatenate
from keras.optimizers import Adam, SGD, Adadelta
from keras import backend as K
from keras.models import Model
from keras.utils import multi_gpu_model
import tensorflow as tf

from keras.layers import normalization, Conv1D,Conv2D
from keras.layers import (Dense, LSTM, Bidirectional,
                          Input, GRU, TimeDistributed)


_CONV_FILTERS = 32
def am_hparams():
    params = tf.contrib.training.HParams(
        # vocab
        vocab_size = 50,
        lr = 0.01,
        gpu_nums = 1,
        is_training = True)
    return params


# =============================搭建模型====================================
class Am():
    """docstring for Amodel."""
    def __init__(self, args):
        self.vocab_size = args.vocab_size
        self.gpu_nums = args.gpu_nums
        self.lr = args.lr
        self.is_training = args.is_training
        self.batch_norm=False
        self._model_init()
        if self.is_training:
            self._ctc_init()
            self.opt_init()

    def _model_init(self):
        #nodes=1000
        gru_hidden=400
        initialization = 'glorot_uniform'
        self.inputs = Input(name='the_inputs', shape=(None, 200, 1))
        #acoustic_input = Reshape((-1, 200))(self.inputs)
        conv_1 = Conv2D(filters=_CONV_FILTERS,kernel_size=(41, 11),strides=(2, 2),name='cnn_1',
               use_bias=False, activation='relu',padding='valid')(self.inputs)
        conv_1_bn = normalization.BatchNormalization(name='bn_conv_1')(conv_1, training=True)

        conv_2 = Conv2D(filters=_CONV_FILTERS,kernel_size=(21, 11),strides=(2, 1),name='cnn_2',
               use_bias=False, activation='relu',padding='valid')(conv_1_bn)
        conv_2_bn = normalization.BatchNormalization(name='bn_conv_2')(conv_2, training=True)
        feat_size = conv_2_bn.get_shape().as_list()[2]
        outputs = Reshape([-1, feat_size * _CONV_FILTERS])(conv_2_bn)

        for r in range(1):
            # output = GRU(nodes, activation='relu',
            #              name='rnn_{}'.format(r + 1), init=initialization,
            #              return_sequences=True)(output)
            outputs = Bidirectional(GRU(gru_hidden, return_sequences=True), name='bi_lstm_{}'.format(r + 1))(outputs)
            if self.batch_norm:
                bn_layer = normalization.BatchNormalization(name='bn_rnn_{}'.format(r + 1),
                                                            moving_mean_initializer='zeros')
                outputs = bn_layer(outputs, training=True)

        self.outputs = TimeDistributed(Dense(
            self.vocab_size, name='dense', activation='softmax', init=initialization,
        ))(outputs)
        self.model = Model(inputs=self.inputs, outputs=self.outputs)
        self.model.summary()

    def _ctc_init(self):
        self.labels = Input(name='the_labels', shape=[None], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')
        self.loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc')\
            ([self.labels, self.outputs, self.input_length, self.label_length])
        self.ctc_model = Model(inputs=[self.labels, self.inputs,
            self.input_length, self.label_length], outputs=self.loss_out)

    def opt_init(self):
        opt = Adam(lr = self.lr, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01, epsilon = 10e-8)
        if self.gpu_nums > 1:
            self.ctc_model=multi_gpu_model(self.ctc_model,gpus=self.gpu_nums)
        self.ctc_model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=opt)





# def dense(units, x, activation="relu"):
#     #x = Dropout(0.2)(x)
#     y = Dense(units, activation=activation, use_bias=True,
#         kernel_initializer='he_normal')(x)
#     return y

def ctc_lambda(args):
    labels, y_pred, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
