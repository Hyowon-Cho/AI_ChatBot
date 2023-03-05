# 가공된 대화 데이터를 기반으로 챗봇 모델을 훈련시키기 위해서는 입력 문장과 대응하는 출력 문장 쌍을 만들어야 한다. 이 예시에서는 입력 문장에 대해 출력 문장을 생성하기 위해 Seq2Seq 모델을 사용한다.
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 랜덤 시드 설정
np.random.seed(0)
tf.random.set_seed(0)

# 문장 최대 길이
MAX_LEN = 30

# 토크나이저 생성
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
texts = []
for pair in qa_pairs:
    texts.append(pair[0])
    texts.append(pair[1])
tokenizer.fit_on_texts(texts)

# 입력 시퀀스 생성
encoder_input_data = []
for pair in qa_pairs:
    input_seq = tokenizer.texts_to_sequences([pair[0]])[0]
    encoder_input_data.append(input_seq)

# 출력 시퀀스 생성
decoder_input_data = []
decoder_target_data = []
for pair in qa_pairs:
    target_seq = tokenizer.texts_to_sequences([pair[1]])[0]
    decoder_input_data.append([tokenizer.word_index["<start>"]] + target_seq)
    decoder_target_data.append(target_seq + [tokenizer.word_index["<end>"]])

# 시퀀스 패딩
encoder_input_data = pad_sequences(encoder_input_data, maxlen=MAX_LEN, padding='post')
decoder_input_data = pad_sequences(decoder_input_data, maxlen=MAX_LEN, padding='post')
decoder_target_data = pad_sequences(decoder_target_data, maxlen=MAX_LEN, padding='post')
