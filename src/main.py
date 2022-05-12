from load_image import load_image, get_total, get_num_of_classification
from collections import OrderedDict
from neural_network import Network
from datetime import date
import cupy as np
import json
import time

data_option = OrderedDict({
    'width': 45,
    'height': 45,
    'validation_ratio': 0.2,
    'classification': {
        'A': {
            'count': 945
        },
        'B': {
            'count': 880
        },
        'C': {
            'count': 883
        },
        'D': {
            'count': 854
        },
        'E': {
            'count': 896
        },
        'K': {
            'count': 922
        },
        'L': {
            'count': 900
        },
        'M': {
            'count': 905
        },
        'N': {
            'count': 891
        },
        'P': {
            'count': 903
        },
        'Q': {
            'count': 894
        },
        'R': {
            'count': 921
        },
        'S': {
            'count': 877
        },
        'T': {
            'count': 860
        },
        'U': {
            'count': 855
        },
        'W': {
            'count': 899
        },
        'X': {
            'count': 945
        },
        'Z': {
            'count': 904
        }
    }
})

# 데이터 로드
(train_img, train_label), (test_img, test_label) = load_image(data_option)

# 학습용 데이터 개수
train_size = train_img.shape[0]

# 미니배치 크기
batch_size = 1000

# 학습률
learning_rate = 0.1

# 입력층 크기
input_size = int(data_option['width'] * data_option['height'])

# 은닉층 크기
hidden_size = 100

# 출력층 크기
output_size = get_num_of_classification(data_option)

# 1 에폭당 반복 수
iter_per_epoch = int(max(train_size / batch_size, 1))

# 반복 횟수
# e.g. 5 => 5 epoch 진행
iters_num = 5


def training():
    # 신경망 객체 생성
    network = Network(input_size, hidden_size, output_size)

    print('===== ===== ===== =====', flush=True)
    print('Number of data : ' + str(get_total(data_option)), flush=True)
    print('Shape of train_img : ' + str(train_img.shape), flush=True)
    print('Shape of train_label : ' + str(train_label.shape), flush=True)
    print('Shape of test_img : ' + str(test_img.shape), flush=True)
    print('Shape of test_label : ' + str(test_label.shape), flush=True)
    print('===== ===== ===== =====', flush=True)

    epoch = 1
    iters_total = iter_per_epoch * iters_num

    # 결과 저장
    file_name = str(date.today()) + '_' + str(int(time.time())) + '.json'

    for i in range(iters_total):
        print('Iteration : ' + str(i + 1) +
              ' / ' + str(iters_total), flush=True)
        print('Epoch : ' + str(epoch) + ' / ' + str(iters_num), flush=True)

        params_to_json = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'iters_num': iters_num,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
        }

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
        print('Loss : ' + str(loss), flush=True)

        batch_accuracy = network.accuracy(batch_img, batch_label)
        print('Batch Accuracy : ' + str(batch_accuracy * 100) + '%', flush=True)

        # 1 epoch당 정확도 계산
        if ((i + 1) % iter_per_epoch == 0):
            print(str(epoch) + ' epoch is over!', flush=True)

            train_accuracy = network.accuracy(train_img, train_label)
            print('Train Accuracy : ' +
                  str(train_accuracy * 100) + '%', flush=True)

            test_accuracy = network.accuracy(test_img, test_label)
            print('Test Accuracy : ' + str(test_accuracy * 100) + '%', flush=True)

            epoch += 1

        print('===== ===== ===== =====', flush=True)


if __name__ == '__main__':
    training()
