# 인코더 모델 생성
encoder_model = Model(encoder_inputs, encoder_states)

# 디코더 모델 생성
decoder_state_input_h = Input(shape=(LSTM_HIDDEN_DIM,))
decoder_state_input_c = Input(shape=(LSTM_HIDDEN_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
dec_emb2= dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)

# 문장 생성 함수
def generate_response(input_text):
    # 입력 문장 전처리
    input_seq = tokenizer.texts_to_sequences([input_text])[0]
    input_seq = pad_sequences([input_seq], maxlen=MAX_LEN, padding='post')
    
    # 입력 문장 인코딩
    states_value = encoder_model.predict(input_seq)
    
    # 디코더 초기 상태 설정
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = None
        for word, index in tokenizer.word_index.items():
            if sampled_token_index == index:
                decoded_sentence += ' {}'.format(word)
                sampled_token = word
        
        # 종료 조건 설정
        if sampled_token == '<end>' or len(decoded_sentence.split()) >= MAX_LEN:
            stop_condition = True
            
        # 디코더 입력 및 상태 갱신
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return decoded_sentence.strip()

# 문장 생성 예시
generate_response('안녕하세요') # ' 안녕하세요.' 또는 ' 만나서 반갑습니다.'와 같은 문장이 출력됩니다.
