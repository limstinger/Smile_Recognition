import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import random

# 현재 작업 디렉토리 가져오기
base_dir = os.getcwd()

# 데이터셋 경로 설정
image_dir = os.path.join(base_dir, 'img_align_celeba')
model_save_path = os.path.join(base_dir, 'models', 'celeba_smiling_model.h5')

# 모델 로드
model = load_model(model_save_path)

# 웃음 강도를 해석하는 함수
def interpret_smiling_probability(prob):
    if prob < 0.2:
        return "Not Smiling"
    elif prob < 0.4:
        return "Barely Smiling"
    elif prob < 0.6:
        return "Moderately Smiling"
    elif prob < 0.8:
        return "Smiling"
    else:
        return "Highly Smiling"

# 이미지 전처리 함수
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    return image.astype(np.float32)

# 이미지 디렉토리에서 임의의 이미지 4개 선택
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
random_image_files = random.sample(image_files, 4)
test_image_paths = [os.path.join(image_dir, img) for img in random_image_files]

# 2x2 그리드로 이미지와 결과를 시각화
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

for i, ax in enumerate(axs.flat):
    test_image_path = test_image_paths[i]
    test_image = preprocess_image(test_image_path)
    test_image = np.expand_dims(test_image, axis=0)
    prediction = model.predict(test_image)
    smiling_probability = prediction[0][0]
    result = interpret_smiling_probability(smiling_probability)  # 수정된 부분: 해석 함수 사용

    # 원본 이미지 로드 및 시각화
    original_image = cv2.imread(test_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    ax.imshow(original_image)
    ax.set_title(f'Smiling Probability: {smiling_probability:.4f}\n{result}')
    ax.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.show()

# 선택된 이미지 경로 출력
for path in test_image_paths:
    print(f'Test image path: {path}')