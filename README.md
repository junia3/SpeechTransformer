# SpeechTransformer
SpeechTransformer

#### 사용한 데이터셋 ####
librispeech_100h, 360h : training
librispeech_dev : validation

링크 : https://www.openslr.org/12

기존 모델에서 VGG extractor를 단순히 mel filterbank로 바꿈.
그리고 sequence 정보를 따로 csv 파일로 생성하여 학습시킴
본래 목적은 streaming KWS까지 진행이었으나 시간 부족으로 간단하게 마무리

#### 사용된 training 조건 ####

임시방편으로 짧은 training을 train.py에 구현하였음.

##### THE SPEECHTRANSFORMER FOR LARGE-SCALE MANDARIN CHINESE SPEECH RECOGNITION #####

라는 논문을 참고함
위의 논문 내용 중에 learning rate를 신기하게 작성한 부분이 있어, 이를 코드로 옮겨서 사용함
