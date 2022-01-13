# 2022_PythonStudy_face_emotion_recognition

사용한 데이터 (kaggle)
1. https://www.kaggle.com/msambare/fer2013 (labeled)
2. https://www.kaggle.com/greatgamedota/ffhq-face-data-set (unlabeled)
3. https://www.kaggle.com/ashwingupta3012/human-faces (unlabeled)

# 과정

1. 데이터 수집 및 A. Vulpe-Grigoraşi and O. Grigore(2021)의 CNN 모형 이용
2. 실행 후 정확도 점검
3. 데이터 증강(좌우반전, 회전) ★
4. 증강(좌우반전, x2)한 데이터로 실행 후 정확도 점검
5. 라벨 없는 데이터 예측 후 예측 확률이 높은(0.95~) 데이터를 라벨링하여 모델 재생성 ★
6. 과정 5) 반복 ★


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


# ★ 데이터(라벨)의 불균형

기존 fer2013은 데이터가 어느정도 균형 (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral. The Disgust expression has the minimal number of images – 600, while other labels have nearly 5,000 samples each.)

하지만 수집한 사진들이 3(Happy)로 예측되는 경우가 많아서 ssl시 예측 시 확률이 높은 데이터의 label이 3인 경우가 많음 << 재학습 시 label = 3 인 경우만 input으로 들어감,,

👍 재학습 데이터를 기존 데이터와 결합을 통해 해결


# MC dropout : test(predict)에서도 적용되는 dropout 적용

Gal, Y., & Ghahramani, Z. (2016, June). Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning (pp. 1050-1059). PMLR.

어렵지만 대충 요약하면 

![image](https://user-images.githubusercontent.com/75729975/149254593-66b4a686-dce6-4083-89a2-fc8d5462c76e.png)

1. softmax로 얻어지는 결과가 실제 확률이나 불확실성을 뜻하지 않는다는 것이다. 예를 들어, 고양이 품종에 관한 모델을 만들었을 때, 학습 때 사용하지 않은 새로운 품종의 고양이나 아니면 고양이가 아닌(강아지) input을 넣었을 때에 올바른 대답은 "모른다"인데, 실제로는 실행했을 때 그렇지 않다. 어떻게서든 아무 라벨을 찾아서 갖다주는데, 이는 softmax가 데이터에 대한 confidence이나 확신을 나타낼 수 없다는 것이다.

2. 이를 구현하기 위해서, dropout을 적용한 신경망 모형들(학습 과정과 데이터는 같은데 random bernoulli에 대해 dropout을 적용하므로 모델 자체는 모두 달라짐)에 특정 input을 집어 넣었을 때 결과 y의 평균mu와 분산sigma^2을 구할수 있을 것이다. 이때의 sigma는 분류결과의 불확정성에 대한 평가지표이다.

3. 기존 dropout은 train 과정에서 적용, test(predict)과정에서는 모든 노드를 사용하여 예측 >>> 불확실성의 검증을 통해 self training의 대상이 되는 낮은 불확실성의 unlabeled data를 고르기 위해서 Monte-Carlo Dropout을 사용하였다. 저장된 같은 모델을 사용해도 매번 예측 결과가 달라지며, 이때의 결과들의 분포는 정규분포에 근사함.


# CNN-RNN 모델 활용?
1. CNN-RNN 모형을 통한 표정 인식; CNN +맥락을 파악할 수 있는 RNN(IRNN)의 장점 이용 => high accuracy
2. 사용한 데이터 : JAFFE, MMI dataset >>> 일반적인 얼굴 사진 X 실험을 위해서 계획되고 규격화된 얼굴 사진임.
3. 눈, 코, 입 이 어느 사진에서든 비슷한 위치에 있을 수 있도록 전처리 + 표준화
4. CNN: 6개의 컨볼루션층과 2개의 fully connected로 이루어진, dropout과 relu activation을 포함한 모델 +weight regularization + adam과 momentum 방식의 업데이트 => 
5. CNN의 용도는 특징 추출
6. RNN: relu activation, CNN의 결과인 feature를 받아서 softmax를 이용한 분류 결과를 냄
7. 모델1: single CNN regression // 모델2: CNN-RNN  ->  모델1보다 모델2의 정확도가 高
모델2는 hyperparameter tuning을 통해서 성능을 높임(히든 유닛과 히든 레이어?)
즉, CNN과 RNN의 결합은 좋은 결과를 보여줄 수 있다.

👎 실제로 해봤을 때는 성능 향상이 거의 없었음(증강 전 데이터 기준)
연구에서 사용한 데이터는 실제로 규격화 및 전처리를 거친 데이터(얼굴의 특정 지점이 거의 비슷한 area에 위치)였기 때문에 RNN이 잘 먹혔던 것 같음.

a. 같은 목적이어도 형식이 다르면 적합한 모델도 달라질 수 있음

b. CNN-RNN을 잘 활용하기 위해서는 얼굴 포인트의 위치를 정렬해야 할 필요가 있다?


# accuracy records
1. cnn_base_model : 0.682
2. cnn_220110_model : 0.687
3. cnn_220111_model : 0.692
4. cnn_220112_model : 0.688
******
5. cnn_base_model_v2 : 0.691
