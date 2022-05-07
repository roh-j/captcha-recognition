from load_image import load_image, get_total, get_num_of_classification
from collections import OrderedDict
from neural_network import Network
from datetime import date
import numpy as np
import json
import time

data_option = OrderedDict({
    'width': 45,
    'height': 45,
    'validation_ratio': 0.2,
    'classification': {
        'A': {
            'count': 35
        },
        'B': {
            'count': 22
        },
        'C': {
            'count': 31
        },
        'D': {
            'count': 32
        },
        'E': {
            'count': 41
        },
        'K': {
            'count': 44
        },
        'L': {
            'count': 44
        },
        'M': {
            'count': 35
        },
        'N': {
            'count': 41
        },
        'P': {
            'count': 27
        },
        'Q': {
            'count': 32
        },
        'R': {
            'count': 37
        },
        'S': {
            'count': 24
        },
        'T': {
            'count': 29
        },
        'U': {
            'count': 22
        },
        'W': {
            'count': 30
        },
        'X': {
            'count': 31
        },
        'Z': {
            'count': 43
        }
    }
})

# 데이터 로드
(train_img, train_label), (test_img, test_label) = load_image(data_option)

# 학습용 데이터 개수
train_size = train_img.shape[0]

# 미니배치 크기
batch_size = 50

# 학습률
learning_rate = 0.5

# 입력 크기
input_size = int(data_option['width'] * data_option['height'])

# 은닉층 크기
hidden_size = 50

# 출력 크기
output_size = get_num_of_classification(data_option)

# 신경망 객체 생성
network = Network(input_size, hidden_size, output_size)

# 1 에폭당 반복 수
iter_per_epoch = int(max(train_size / batch_size, 1))

# 반복 횟수
iters_num = iter_per_epoch * 5


def training():
    print('===== ===== ===== =====')
    print('Number of Data : ' + str(get_total(data_option)))
    print('Shape of train_img : ' + str(train_img.shape))
    print('Shape of train_label : ' + str(train_label.shape))
    print('Shape of test_img : ' + str(test_img.shape))
    print('Shape of test_label : ' + str(test_label.shape))
    print('===== ===== ===== =====')

    epoch = 1
    total_epoch = int(iters_num / iter_per_epoch)
    file_name = str(date.today()) + '_' + str(int(time.time())) + '.json'

    for i in range(iters_num):
        print('iteration : ' + str(i + 1) + ' / ' + str(iters_num))
        print('epoch : ' + str(epoch) + ' / ' + str(total_epoch))

        params_to_json = {}

        # 미니배치
        batch_mask = np.random.choice(train_size, batch_size)

        batch_img = train_img[batch_mask]
        batch_label = train_label[batch_mask]

        # 기울기 계산
        gradient = network.numerical_gradient(batch_img, batch_label)

        # 학습
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * gradient[key]
            params_to_json[key] = network.params[key].tolist()

        # 학습 결과 저장
        with open('result/' + file_name, 'w') as file:
            learned_params = json.dumps(params_to_json, indent=4)
            file.write(learned_params)

        # 결과 기록
        loss = network.loss(batch_img, batch_label)
        print('Loss : ' + str(loss))

        batch_accuracy = network.accuracy(batch_img, batch_label)
        print('Batch Accuracy : ' + str(batch_accuracy * 100) + '%')

        # 1 epoch당 정확도 계산
        if ((i + 1) % iter_per_epoch == 0):
            print(str(epoch) + ' epoch is over!')

            train_accuracy = network.accuracy(train_img, train_label)
            print('Train Accuracy : ' + str(train_accuracy * 100) + '%')

            test_accuracy = network.accuracy(test_img, test_label)
            print('Test Accuracy : ' + str(test_accuracy * 100) + '%')

            epoch += 1

        print('===== ===== ===== =====')


if __name__ == '__main__':
    training()
