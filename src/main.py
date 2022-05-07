from load_image import load_image, get_total, get_num_of_classification
from collections import OrderedDict
from neural_network import Network
import numpy as np

data_option = OrderedDict({
    'width': 28,
    'height': 28,
    'validation_ratio': 0.5,
    'classification': {
        '0': {
            'count': 2
        },
        '1': {
            'count': 2
        }
    }
})

# 데이터 로드
(train_img, train_label), (test_img, test_label) = load_image(data_option)

# 손실 값 추이
loss_history = []

# 반복 횟수
iters_num = 1000

# 학습용 데이터 개수
train_size = train_img.shape[0]

# 미니배치 크기
batch_size = 100

# 학습률
learning_rate = 0.1

# 입력 크기
input_size = int(data_option['width'] * data_option['height'])

# 은닉층 크기
hidden_size = 50

# 출력 크기
output_size = get_num_of_classification(data_option)

# 신경망 객체 생성
network = Network(input_size, hidden_size, output_size)


def training():
    print('Number of Data : ' + str(get_total(data_option)))

    for i in range(iters_num):
        # 미니배치
        batch_mask = np.random.choice(train_size, batch_size)

        batch_img = train_img[batch_mask]
        batch_label = train_img[batch_mask]

        # 기울기 계산
        gradient = network.numerical_gradient(batch_img, batch_label)

        # 학습
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * gradient[key]

        # 결과 기록
        loss = network.loss(batch_img, batch_label)
        loss_history.append(loss)
