import keras
from keras.layers import Input, BatchNormalization, LSTM
from keras.layers import Reshape, Dense, Lambda, Dropout
from keras.layers.recurrent import GRU
from keras.layers.merge import add, concatenate
from keras.optimizers import Adam, SGD, Adadelta
from keras import backend as K
from keras.models import Model
from keras.utils import multi_gpu_model
import tensorflow as tf

from keras.layers import normalization, Conv1D
from keras.layers import (Convolution1D, Dense, LSTM, Bidirectional,
                          Input, GRU, TimeDistributed)

def am_hparams():
    params = tf.contrib.training.HParams(
        # vocab
        vocab_size = 50,
        lr = 0.0008,
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
        nodes=1000
        initialization = 'glorot_uniform'
        self.inputs = Input(name='the_inputs', shape=(None, 200, 1))
        acoustic_input = Reshape((-1, 200))(self.inputs)
        conv_1d = Convolution1D(nodes, 11, name='conv1d',
                                border_mode='valid',
                                subsample_length=2, init=initialization,
                                activation='relu')(acoustic_input)
        if self.batch_norm:
            output = normalization.BatchNormalization(name='bn_conv_1d')(conv_1d, training=True)
        else:
            output = conv_1d

        for r in range(3):
            # output = GRU(nodes, activation='relu',
            #              name='rnn_{}'.format(r + 1), init=initialization,
            #              return_sequences=True)(output)
            output = Bidirectional(GRU(nodes, return_sequences=True), name='bi_lstm_{}'.format(r + 1))(output)
            if self.batch_norm:
                bn_layer = normalization.BatchNormalization(name='bn_rnn_{}'.format(r + 1),
                                                            moving_mean_initializer='zeros')
                output = bn_layer(output, training=True)

        self.outputs = TimeDistributed(Dense(
            self.vocab_size, name='dense', activation='softmax', init=initialization,
        ))(output)
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




# ============================模型组件=================================
def bi_gru(units, x):
    x = Dropout(0.2)(x)
    y1 = GRU(units, return_sequences=True,
        kernel_initializer='he_normal')(x)
    y2 = GRU(units, return_sequences=True, go_backwards=True,
        kernel_initializer='he_normal')(x)
    y = add([y1, y2])
    return y


def dense(units, x, activation="relu"):
    x = Dropout(0.2)(x)
    y = Dense(units, activation=activation, use_bias=True,
        kernel_initializer='he_normal')(x)
    return y

def ctc_lambda(args):
    labels, y_pred, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
