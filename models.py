import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2

# 현재 작업 디렉토리 가져오기
base_dir = os.getcwd()

# 데이터셋 경로 설정
image_dir = os.path.join(base_dir, 'img_align_celeba')
attributes_path = os.path.join(base_dir, 'Anno', 'list_attr_celeba.txt')
partition_path = os.path.join(base_dir, 'Eval', 'list_eval_partition.txt')

# 속성 데이터 로드
with open(attributes_path, 'r') as file:
    lines = file.readlines()

# 첫 번째 줄은 헤더, 두 번째 줄은 설명, 그 이후는 데이터
columns = lines[1].strip().split()
data = [line.strip().split() for line in lines[2:]]

# 데이터프레임으로 변환
attributes = pd.DataFrame(data, columns=['image_id'] + columns)
attributes.set_index('image_id', inplace=True)
attributes = attributes.apply(pd.to_numeric)

# -1 값을 0으로 변환
attributes.replace(-1, 0, inplace=True)

# 평가 데이터 로드
partitions = pd.read_csv(partition_path, delim_whitespace=True, header=None, index_col=0)
partitions.columns = ['partition']

# 이미지 전처리 함수
def preprocess_image(image_path):
    image_path_str = image_path.decode('utf-8')
    image = cv2.imread(image_path_str)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    return image.astype(np.float32)  # float32로 변환

def load_and_preprocess_image(path, label):
    image = tf.numpy_function(preprocess_image, [path], tf.float32)
    image.set_shape((128, 128, 3))
    return image, label

# 이미지 경로와 레이블을 텐서플로우 데이터셋으로 변환
image_paths = [os.path.join(image_dir, img) for img in attributes.index]
labels = attributes['Smiling'].values

# 데이터셋 분할
train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2, stratify=labels, random_state=42)
val_paths, test_paths, val_labels, test_labels = train_test_split(test_paths, test_labels, test_size=0.5, stratify=test_labels, random_state=42)

# 텐서플로우 데이터셋 생성
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

# 데이터셋 전처리 및 배치 설정
BATCH_SIZE = 32
train_ds = train_ds.map(lambda path, label: (load_and_preprocess_image(path, label))).shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.map(lambda path, label: (load_and_preprocess_image(path, label))).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.map(lambda path, label: (load_and_preprocess_image(path, label))).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# 모델 설계
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 요약
model.summary()

# 모델 훈련
EPOCHS = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# 모델 평가
loss, accuracy = model.evaluate(test_ds)
print(f'Test accuracy: {accuracy:.4f}')

# 모델 저장 경로 설정
model_save_path = os.path.join(base_dir, 'models', 'celeba_smiling_model.h5')
model.save(model_save_path)