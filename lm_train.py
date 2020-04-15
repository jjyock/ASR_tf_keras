

#2.语言模型训练-------------------------------------------
import tensorflow as tf
from model_language.transformer import Lm, lm_hparams
from am_train import thchs30,aishell,prime,stcmd,batch_size,data_length,is_shuffle
from utils import data_hparams,get_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm
if __name__ == '__main__':


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
    lm_args = lm_hparams()
    lm_epochs = 10

    lm_args.num_heads = 8
    lm_args.num_blocks = 6
    lm_args.input_vocab_size = len(train_data.pny_vocab)
    lm_args.label_vocab_size = len(train_data.han_vocab)
    lm_args.max_length = 100
    lm_args.hidden_units = 512
    lm_args.dropout_rate = 0.2
    lm_args.lr = 0.0003
    lm_args.is_training = True
    lm = Lm(lm_args)
    batch_num = len(train_data.wav_lst) // train_data.batch_size

    with lm.graph.as_default():
        saver =tf.train.Saver()
    with tf.Session(graph=lm.graph) as sess:
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        add_num = 0
        # if os.path.exists('logs_lm/checkpoint'):
        #     print('loading language model...')
        #     latest = tf.train.latest_checkpoint('logs_lm')
        #     add_num = int(latest.split('_')[-1])
        #     saver.restore(sess, latest)
        #writer = tf.summary.FileWriter('logs_lm/tensorboard', tf.get_default_graph())
        for k in range(lm_epochs):
            total_loss = 0
            batch = train_data.get_lm_batch()
            for i in tqdm(range(batch_num)):
                input_batch, label_batch = next(batch)
                feed = {lm.x: input_batch, lm.y: label_batch}
                cost,_ = sess.run([lm.mean_loss,lm.train_op], feed_dict=feed)
                total_loss += cost
                if (k * batch_num + i) % 10 == 0:
                    rs=sess.run(merged, feed_dict=feed)
                    #writer.add_summary(rs, k * batch_num + i)
            print('epochs', k+1, ': average loss = ', total_loss/batch_num)
        saver.save(sess, 'logs_lm/model_%d' % (lm_epochs + add_num))
        #writer.close()