import os
import tensorflow as tf
from utils import get_data, data_hparams
from keras.callbacks import ModelCheckpoint
from utils import decode_ctc, GetEditDistance
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.logging.set_verbosity(tf.logging.ERROR)
# dataset to use
thchs30 = True
stcmd = True
prime = True
aishell = True
batch_size = 10
data_length = None
is_shuffle = True
am_epochs = 10
use_pretrain = True
model_save_path = 'logs_am/model_deepspeech.h5'

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
    print(train_data)
    #from model_speech.cnn_ctc import Am, am_hparams
    from model_speech.deepspeech_new import Am,am_hparams
    am_args = am_hparams()
    am_args.vocab_size = len(train_data.am_vocab)
    am_args.gpu_nums = 1
    am_args.lr = 0.0008
    am_args.is_training = True
    am = Am(am_args)

    if os.path.exists(model_save_path) and use_pretrain:
        print('load acoustic model...')
        am.ctc_model.load_weights(model_save_path)

    batch_num = len(train_data.wav_lst) // train_data.batch_size
    #
    # # checkpoint
    ckpt = "model_{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(
        os.path.join(
            './checkpoint',
            ckpt),
        monitor='val_loss',
        save_weights_only=False,
        verbose=1,
        save_best_only=True)
    for k in range(am_epochs):
        print('this is the', k + 1, 'th epochs trainning !!!')
        train_batch = train_data.get_am_batch()
        am.ctc_model.fit_generator(
            train_batch,
            steps_per_epoch=batch_num,
            epochs=1,
            workers=1,
        )


        #dev_batch = dev_data.get_am_batch()
        #am.ctc_model.fit_generator(train_batch, steps_per_epoch=batch_num, epochs=1, callbacks=[checkpoint], workers=1, use_multiprocessing=False, validation_data=dev_batch, validation_steps=200)
        am.ctc_model.save_weights(model_save_path)
