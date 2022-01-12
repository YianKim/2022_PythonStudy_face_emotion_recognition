# 2022_PythonStudy_face_emotion_recognition

사용한 데이터 (kaggle)
1. https://www.kaggle.com/msambare/fer2013 (labeled)
2. https://www.kaggle.com/greatgamedota/ffhq-face-data-set (unlabeled)
3. https://www.kaggle.com/ashwingupta3012/human-faces (unlabeled)

# 과정

1. 데이터 수집 및 kaggle code의 cnn 모형 코드 이용
2. 실행 후 정확도 점검(62~%)
3. 데이터 증강(좌우반전, 회전) ★
4. 증강한 데이터로 실행 후 정확도 점검(x2, 68~%)
5. 라벨 없는 데이터 예측 후 예측 확률이 높은(0.95~) 데이터를 라벨링하여 모델 재생성 ★
6. 과정 5 반복 ★


# ★ unlabeled 데이터 전처리

![image](https://user-images.githubusercontent.com/75729975/148866954-5914f5fa-7875-4b4b-bab7-b464df7c8c6d.png)


# ★ 준지도 pseudo labeling : self supervised

![image](https://user-images.githubusercontent.com/75729975/148865800-8efde425-daba-41ae-95d9-110f6a51569f.png)

1. fer2013 + augmentation을 통해 CNN 학습 및 저장
2. 예측 확률 0.95 이상의 데이터를 추출
3. (fer2013 + augmentation) + 2. 의 데이터를 통해 1.의 CNN 추가 학습
4. 이 때, 학습률은 1.에서 보다 작게 실행 (기존 학습이 매몰되지 않도록)
5. 3번의 모델을 저장
6. 과정 2~4 을 반복

재학습을 통해서 test set의 정확도 소폭 향상 >>> 결과가 0.95가 넘는 데이터 새로 생성

# accuracy records
1. cnn_base_model : 0.682
2. cnn_220110_model : 0.687
3. cnn_220111_model : 0.692
