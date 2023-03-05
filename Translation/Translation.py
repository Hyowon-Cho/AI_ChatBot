# Google Cloud Translation API를 사용

# Google Cloud Platform Console에서 새 프로젝트를 생성함.
# Cloud Translation API를 활성화함.
# 새로운 인증서비스 계정을 만들고 해당 서비스 계정의 키를 JSON 형식으로 다운로드함.
# 다음과 같이 google-auth와 google-cloud-translate 패키지를 설치. : !pip install google-auth google-auth-oauthlib google-auth-httplib2 google-cloud-translate

# 번역 API 구현

import os
from google.oauth2 import service_account
from google.cloud import translate_v2 as translate

# Google Cloud Translation API 설정
key_path = 'google_translate_key.json'
credentials = service_account.Credentials.from_service_account_file(key_path)
translate_client = translate.Client(credentials=credentials)

# 문장 번역 함수
def translate_text(text, target_language):
    # 입력 문장 번역
    result = translate_client.translate(text, target_language=target_language)
    translated_text = result['translatedText']
    return translated_text

# 문장 생성 함수 (번역 기능 추가)
def generate_response(input_text, target_language):
    # 입력 문장 번역
    input_text = translate_text(input_text, 'en')
    
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
    
    # 출력 문장 번역
    decoded_sentence = translate_text(decoded_sentence.strip(), target_language)
    return decoded_sentence
