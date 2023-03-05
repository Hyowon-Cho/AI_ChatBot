from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model

# 임베딩 차원과 LSTM 은닉 상태 차원 설정
EMBEDDING_DIM = 256
LSTM_HIDDEN_DIM = 256

# 인코더 설정
encoder_inputs = Input(shape=(MAX_LEN,))
enc_emb = Embedding(len(tokenizer.word_index)+1, EMBEDDING_DIM)(encoder_inputs)
encoder_lstm = LSTM(LSTM_HIDDEN_DIM, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# 디코더 설정
decoder_inputs = Input(shape=(MAX_LEN,))
dec_emb_layer = Embedding(len(tokenizer.word_index)+1, E

# 디코더 LSTM 설정
decoder_lstm = LSTM(LSTM_HIDDEN_DIM, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

# 출력층 설정
decoder_dense = Dense(len(tokenizer.word_index)+1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# 모델 설정
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 모델 컴파일
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])

# 모델 훈련
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, epochs=50, batch_size=64, validation_split=0.2)
