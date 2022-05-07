import cv2
import numpy as np


def load_image(data_option):
    '''
    이미지 데이터를 불러오는 함수
    '''

    classification = data_option['classification']
    num_of_classification = len(classification.keys())

    # 이미지 2차원을 1차원으로 변경
    flat_dim = int(data_option['width'] * data_option['height'])

    '''
    학습용 데이터
    '''
    # 2차원 형태로 행에 append 하기 위해서는 차원이 지정되어 있어야 함
    train_img = np.empty((0, flat_dim), dtype=int)
    # One-hot encoding 적용
    train_label = np.empty((0, num_of_classification), dtype=int)

    '''
    Validation용 데이터
    '''
    test_img = np.empty((0, flat_dim), dtype=int)
    test_label = np.empty((0, num_of_classification), dtype=int)

    for key in classification:
        count = classification[key]['count']
        validation_mask = np.random.choice(
            count, int(count * data_option['validation_ratio']))

        for i in range(count):
            img_src = 'dataset/img/' + key + '/' + str(i + 1) + '.png'

            # 28 x 28 크기 이미지를 Gray Scale로 가져옴
            img = cv2.imread(img_src, cv2.IMREAD_GRAYSCALE)

            # 2차원 행렬을 1차원으로 변경
            # (28, 28) => (784, )
            img = img.reshape(-1)

            target = np.zeros(num_of_classification, dtype=int)
            target[list(classification.keys()).index(key)] = 1

            # Validation용 데이터로 저장
            if (i in validation_mask):
                test_img = np.append(test_img, np.array([img]), axis=0)
                test_label = np.append(
                    test_label, np.array([target]), axis=0)

            # 학습용 데이터로 저장
            else:
                train_img = np.append(train_img, np.array([img]), axis=0)
                train_label = np.append(
                    train_label, np.array([target]), axis=0)

    return [(train_img, train_label), (test_img, test_label)]


def get_total(data_option):
    '''
    데이터 총개수를 반환하는 함수
    '''

    classification = data_option['classification']
    total = 0

    for key in classification:
        total += classification[key]['count']

    return total


def get_num_of_classification(data_option):
    '''
    분류 개수를 반환하는 함수
    '''

    classification = data_option['classification']

    return len(classification.keys())
