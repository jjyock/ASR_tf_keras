
import difflib
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from tqdm import tqdm
from scipy.fftpack import fft
from python_speech_features import mfcc
from random import shuffle
from keras import backend as K
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))




class get_data():
    def __init__(self, args):
        self.data_type = args.data_type
        self.data_path = args.data_path
        self.thchs30 = args.thchs30
        self.aishell = args.aishell
        self.prime = args.prime
        self.stcmd = args.stcmd
        self.data_length = args.data_length
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.dict_dir = os.path.join(PROJECT_PATH,'dict')
        self.task_id = args.task_id
        if not os.path.exists(self.dict_dir):
            os.mkdir(self.dict_dir)
        self.source_init()

    def source_init(self):
        print('get source list...')
        read_files = []
        if self.data_type == 'train':
            if self.thchs30 == True:
                read_files.append('thchs_train.txt')
            if self.aishell == True:
                read_files.append('aishell_train.txt')
            if self.prime == True:
                read_files.append('prime.txt')
            if self.stcmd == True:
                read_files.append('stcmd_train.txt')
        elif self.data_type == 'dev':
            if self.thchs30 == True:
                read_files.append('thchs_dev.txt')
            if self.aishell == True:
                read_files.append('aishell_dev.txt')
            if self.stcmd == True:
                read_files.append('stcmd_dev.txt')
        elif self.data_type == 'test':
            if self.thchs30 == True:
                read_files.append('thchs_test.txt')
            if self.aishell == True:
                read_files.append('aishell_test.txt')
            if self.stcmd == True:
                read_files.append('stcmd_test.txt')
        self.wav_lst = []
        self.pny_lst = []
        self.han_lst = []
        for file in read_files:
            print('load ', file, ' data...')
            sub_file = os.path.join(self.data_path , file)
            with open(sub_file, 'r', encoding='utf8') as f:
                data = f.readlines()
            for line in tqdm(data):
                wav_file, pny, han = line.split('\t')
                self.wav_lst.append(wav_file)
                self.pny_lst.append(pny.split(' '))
                self.han_lst.append(han.strip('\n'))
        if self.data_length:
            self.wav_lst = self.wav_lst[:self.data_length]
            self.pny_lst = self.pny_lst[:self.data_length]
            self.han_lst = self.han_lst[:self.data_length]
        print('make am vocab...')
        self.am_vocab = self.mk_am_vocab(self.pny_lst)
        print('make lm pinyin vocab...')
        self.pny_vocab = self.mk_lm_pny_vocab(self.pny_lst)
        print('make lm hanzi vocab...')
        self.han_vocab = self.mk_lm_han_vocab(self.han_lst)
    def mk_am_vocab(self, data):
        vocab_path = os.path.join(self.dict_dir, 'am_vocab_%d.joblib' % (self.task_id))

        if self.data_type=='train':
            vocab = sorted(list(set([y for x in data for y in x]))) + ['_']

            print('am vocab num is %d' % (len(vocab)))

            joblib.dump(vocab,vocab_path)
        else:
            vocab = joblib.load(vocab_path)
            print('load am vocab num is %d'%(len(vocab)))

        return vocab

    def mk_lm_pny_vocab(self, data):
        vocab_path = os.path.join(self.dict_dir, 'pny_vocab_%d.joblib' % (self.task_id))
        if self.data_type=='train':

            vocab = sorted(list(set([y for x in data for y in x ])))+['<PAD>']
            joblib.dump(vocab,vocab_path)

            print('%s lm pny vocab num is %d' % (self.data_type,len(vocab)))

        else:
            vocab = joblib.load(vocab_path)
            print('lm pny vocab num is %d' % (len(vocab)))

        return vocab

    def mk_lm_han_vocab(self, data):
        vocab_path = os.path.join(self.dict_dir, 'han_vocab_%d.joblib' % (self.task_id))
        if self.data_type=='train':

            vocab = sorted(list(set([y for x in data for y in x.replace(' ',' ') ])))+['<PAD>']
            joblib.dump(vocab,vocab_path)

            print('%s lm han vocab num is %d' % (self.data_type,len(vocab)))
        else:
            vocab = joblib.load(vocab_path)
            print('%s lm han vocab num is %d' % (self.data_type,len(vocab)))


        return vocab
