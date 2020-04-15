import os
import tensorflow as tf
from utils import get_data, data_hparams
from keras.callbacks import ModelCheckpoint
from utils import decode_ctc, GetEditDistance

# dataset to use
thchs30=True
stcmd=False
prime=False
aishell=False
batch_size=10
data_length =100
is_shuffle = True
am_epochs = 1
use_pretrain=False

data_args = data_hparams()

data_args.thchs30 = thchs30
data_args.aishell = aishell
data_args.prime = prime
data_args.stcmd = stcmd
data_args.batch_size = batch_size
data_args.data_length = data_length
data_args.shuffle = is_shuffle

# 0.准备训练所需数据------------------------------

data_args.data_type = 'train'
train_data = get_data(data_args)
print(train_data)
# 0.准备验证所需数据------------------------------


data_args.data_type = 'train'
dev_data = get_data(data_args)
print(dev_data)

# 1.声学模型训练-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams
am_args = am_hparams()
am_args.vocab_size = len(train_data.am_vocab)
am_args.gpu_nums = 1
am_args.lr = 0.0008
am_args.is_training = True
am = Am(am_args)

if os.path.exists('logs_am/model.h5') and use_pretrain:
    print('load acoustic model...')
    am.ctc_model.load_weights('logs_am/model.h5')

batch_num = len(train_data.wav_lst) // train_data.batch_size
#
# # checkpoint
ckpt = "model_{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(os.path.join('./checkpoint', ckpt), monitor='val_loss', save_weights_only=False, verbose=1, save_best_only=True)
#
# #
for k in range(am_epochs):
    print('this is the', k+1, 'th epochs trainning !!!')
    train_batch = train_data.get_am_batch()
    dev_batch = dev_data.get_am_batch()
    am.ctc_model.fit_generator(train_batch, steps_per_epoch=batch_num, epochs=1, callbacks=[checkpoint], workers=1, use_multiprocessing=False, validation_data=dev_batch, validation_steps=200)

    am.ctc_model.save_weights('logs_am/model.h5')

data_args.batch_size=1
test_data = get_data(data_args)
word_num = 0
word_error_num = 0
am_batch= test_data.get_am_batch()
for i in range(10):
    print('\n the ', i, 'th example.')
    # 载入训练好的模型，并进行识别
    inputs, _ = next(am_batch)
    x = inputs['the_inputs']
    y = test_data.pny_lst[i]
    result = am.model.predict(x, steps=1)
    # 将数字结果转化为文本结果
    _, text = decode_ctc(result, train_data.am_vocab)
    text = ' '.join(text)
    print('文本结果：', text)
    print('原文结果：', ' '.join(y))
# print('start language model')
# # 2.语言模型训练-------------------------------------------
# from model_language.transformer import Lm, lm_hparams
# lm_args = lm_hparams()
# lm_args.num_heads = 8
# lm_args.num_blocks = 6
# lm_args.input_vocab_size = len(train_data.pny_vocab)
# lm_args.label_vocab_size = len(train_data.han_vocab)
# lm_args.max_length = 100
# lm_args.hidden_units = 512
# lm_args.dropout_rate = 0.2
# lm_args.lr = 0.0003
# lm_args.is_training = True
# lm = Lm(lm_args)
#
# epochs = 20
# with lm.graph.as_default():
#     saver =tf.train.Saver()
# with tf.Session(graph=lm.graph) as sess:
#     merged = tf.summary.merge_all()
#     sess.run(tf.global_variables_initializer())
#     add_num = 0
#     if os.path.exists('logs_lm/checkpoint'):
#         print('loading language model...')
#         latest = tf.train.latest_checkpoint('logs_lm')
#         add_num = int(latest.split('_')[-1])
#         saver.restore(sess, latest)
#     writer = tf.summary.FileWriter('logs_lm/tensorboard', tf.get_default_graph())
#     for k in range(epochs):
#         total_loss = 0
#         batch = train_data.get_lm_batch()
#         for i in range(batch_num):
#             input_batch, label_batch = next(batch)
#             feed = {lm.x: input_batch, lm.y: label_batch}
#             cost,_ = sess.run([lm.mean_loss,lm.train_op], feed_dict=feed)
#             total_loss += cost
#             if (k * batch_num + i) % 10 == 0:
#                 rs=sess.run(merged, feed_dict=feed)
#                 writer.add_summary(rs, k * batch_num + i)
#         print('epochs', k+1, ': average loss = ', total_loss/batch_num)
#     saver.save(sess, 'logs_lm/model_%d' % (epochs + add_num))
#     writer.close()
