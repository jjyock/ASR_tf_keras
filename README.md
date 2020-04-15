# chinese ASR recognition


reference :

https://github.com/rolczynski/Automatic-Speech-Recognition

https://github.com/audier/DeepSpeechRecognition


baidu deepspeech2:

https://github.com/mozilla/DeepSpeech

https://github.com/tensorflow/models/blob/master/research/deep_speech/deep_speech.py

## 1. Introduction

## 2. acoustics model
deepspeech_new.py :baidu deepspeech2 model

ctc loss based model

## 3. language model


## 4. dataset
stc、primewords、Aishell、thchs30 four datasets about 430 hours.
[http://www.openslr.org/resources.php](http://www.openslr.org/resources.php)


## 5. config

使用train.py文件进行模型的训练。

声学模型可选cnn-ctc、gru-ctc，只需修改导入路径即可：

`from model_speech.cnn_ctc import Am, am_hparams`

`from model_speech.gru_ctc import Am, am_hparams`

语言模型可选transformer和cbhg:

`from model_language.transformer import Lm, lm_hparams`

`from model_language.cbhg import Lm, lm_hparams`
