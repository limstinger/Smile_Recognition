# Code description for models.py

## 경로 설정 
```python
base_dir = os.getcwd()

image_dir = os.path.join(base_dir, 'img_align_celeba')
attributes_path = os.path.join(base_dir, 'Anno', 'list_attr_celeba.txt')
partition_path = os.path.join(base_dir, 'Eval', 'list_eval_partition.txt')
```
* 현재 작업 디렉토리를 가져오고, 이미지 및 속성 데이터, 평가 데이터의 경로를 설정

## 속성 데이터 및 평가 데이터 로드
```python
with open(attributes_path, 'r') as file:
    lines = file.readlines()

columns = lines[1].strip().split()
data = [line.strip().split() for line in lines[2:]]
attributes = pd.DataFrame(data, columns=['image_id'] + columns)
attributes.set_index('image_id', inplace=True)
attributes = attributes.apply(pd.to_numeric)
attributes.replace(-1, 0, inplace=True)

partitions = pd.read_csv(partition_path, delim_whitespace=True, header=None, index_col=0)
partitions.columns = ['partition']
```
* 속성 데이터 및 평가 데이터를 로드하고, 데이터프레임으로 변경
  * 첫 번째 줄은 헤더, 두 번째 줄은 설명, 나머지는 데이터로 처리
  * '-1'값을 '0'으로 치환

## 이미지 전처리 함수
```python
def preprocess_image(image_path):
    image_path_str = image_path.decode('utf-8')
    image = cv2.imread(image_path_str)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    return image.astype(np.float32)
```
* 이미지를 읽고, RGB로 변환, 크기를 조정 후, 정규화하여 반환

## 텐서플로우 데이터셋 생성 함수
```python
def load_and_preprocess_image(path, label):
    image = tf.numpy_function(preprocess_image, [path], tf.float32)
    image.set_shape((128, 128, 3))
    return image, label
```
* 텐서플로우를 사용하기 위해 이미지 전처리 및 로드하는 함수

## 데이터셋 분할 및 텐서플로우 데이터셋 생성
```python
train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2, stratify=labels, random_state=42)
val_paths, test_paths, val_labels, test_labels = train_test_split(test_paths, test_labels, test_size=0.5, stratify=test_labels, random_state=42)
```
* 'train_test_split' 함수를 사용하여 데이터를 train, test, val 세트로 분할
```python
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

BATCH_SIZE = 32
train_ds = train_ds.map(lambda path, label: (load_and_preprocess_image(path, label))).shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.map(lambda path, label: (load_and_preprocess_image(path, label))).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.map(lambda path, label: (load_and_preprocess_image(path, label))).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
```
* 위의 train, test, val 세트를 사용하여 텐서플로우 데이터셋을 생성하고, 배치 처리 및 전처리를 설정
  
## 모델 설정

### CNN 모델 설계
```python
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
```
* 'Sequential' : Keras에서 제공하는 모델 유형 중 하나로, 층을 순차적으로 쌓아 올리는 방식
* 'Conv2D' : 2D 컨볼루션 레이어를 추가
* 'MaxPooling2D((2*2))' : 2x2 최대 풀링 레이어를 추가
  * 해당 함수는 계산량을 줄이고 특징의 추출을 도움
* 'Flatten' : 다차원 배열을 1차원으로 펼침 -> 이는 Dense 레이어에 연결하기 위해 사용
* 'Dense' : 완전 연결(Dense) 레이어
* 'Dense(1, activation='sigmoid')' : 이진 분류 문제(웃고 있는지 아닌지)에 대해 적합

### 모델 컴파일
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
* 'loss='binary_crossentropy' : 이진 분류 문제에 적합한 손실 함수
* metrics=['accuracy'] : 모델의 성능을 정확도로 측정

### 모델 훈련
```python
EPOCHS = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
```

* EPOCHS의 수를 조절하여 모델의 성능을 설정

### 모델 저장
```python
model_save_path = os.path.join(base_dir, 'models', 'celeba_smiling_model.h5')
model.save(model_save_path)
```
* 모델을 지정된 경로에 저장


