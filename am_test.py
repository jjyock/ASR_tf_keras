#coding=utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import difflib
import tensorflow as tf
import numpy as np
from utils import decode_ctc, GetEditDistance,PROJECT_PATH
import joblib

# 0.准备解码所需字典，参数需和训练一致，也可以将字典保存到本地，直接进行读取
from utils import get_data, data_hparams
from am_train import thchs30,aishell,prime,stcmd
data_args = data_hparams()

data_args.thchs30 = thchs30
data_args.aishell = aishell
data_args.prime = prime
data_args.stcmd = stcmd
data_args.data_length = None



# 1.声学模型-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams

am_vocab = joblib.load(os.path.join(os.path.join(PROJECT_PATH,'dict'),'am_vocab_1.joblib'))
pny_vocab = joblib.load(os.path.join(os.path.join(PROJECT_PATH,'dict'),'pny_vocab_1.joblib'))
han_vocab = joblib.load(os.path.join(os.path.join(PROJECT_PATH,'dict'),'han_vocab_1.joblib'))

am_args = am_hparams()
am_args.vocab_size = len(am_vocab)
am = Am(am_args)
print('loading acoustic model...')
am.ctc_model.load_weights('logs_am/model.h5')

data_args.data_type = 'test'
data_args.shuffle = False
data_args.batch_size = 1
test_data = get_data(data_args)

# 4. 进行测试-------------------------------------------
am_batch = test_data.get_am_batch()
word_num = 0
word_error_num = 0
for i in range(1):
    print('\n the ', i, 'th example.')
    # 载入训练好的模型，并进行识别
    inputs, _ = next(am_batch)
    x = inputs['the_inputs']
    y = test_data.pny_lst[i]
    result = am.model.predict(x, steps=1)
    # 将数字结果转化为文本结果
    _, text = decode_ctc(result, am_vocab)
    text = ' '.join(text)
    print('文本结果：', text)
    print('原文结果：', ' '.join(y))

    word_error_num += min(len(y), GetEditDistance(y, text.split(' ')))
    word_num += len(y)
print('词错误率：', word_error_num / word_num)