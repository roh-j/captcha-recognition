import numpy as np


def sigmoid(x):
    '''
    시그모이드 (활성 함수)
    '''

    # x는 numpy arry로 들어오지만 numpy의 브로드캐스트 기능으로 연산 수행 가능
    return 1 / (1 + np.exp(-x))


def softmax(x):
    '''
    결과 값을 0 ~ 1로 정규화 하는 함수
    '''

    # 지수 함수가 포함되어 있기 때문에 지수에 큰 수가 들어가면 연산이 불안정해짐
    # 지수에 최대 값을 빼서 지수 크기를 작게 만듦
    prevention = np.max(x)  # 오버플로우 방지
    numerator = np.exp(x - prevention)
    return numerator / np.sum(numerator)


def mean_squared_error(predict_label, answer_label):
    '''
    평균 제곱 오차 (손실 함수)
    '''

    # predict_label: softmax 결과값으로 0 ~ 1로 정규화된 값
    # answer_label: 정답 레이블 값으로 one-hot 인코딩이 되어 있음
    return 0.5 * np.sum((predict_label - answer_label)**2)


def numerical_gradient(loss, x):
    '''
    편미분 함수 (수치 미분으로 근사)
    오차를 줄이기 위해 중앙 차분 연산
    '''

    h = 1e-4  # 0.0001

    # x가 참조 하고 있는 numpy 배열은 shape 변하지 않음
    shape = x.shape
    x = x.reshape(-1)  # 1차원으로 변경

    # 각 x에 대한 편미분 값
    # x와 형상이 같음
    gradient = np.zeros_like(x)

    # numpy size는 원소 수를 반환함
    for i in range(x.size):
        origin_x = x[i]

        # f(x+h) 함수 계산
        x[i] = origin_x + h
        fxh1 = loss()

        # f(x-h) 함수 계산
        x[i] = origin_x - h
        fxh2 = loss()

        # 값 복원
        x[i] = origin_x

        gradient[i] = (fxh1 - fxh2) / (2 * h)  # 중앙 차분

    # 차원 복원
    gradient = gradient.reshape(shape)

    return gradient


class Network:
    '''
    2층 신경망
    '''

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        '''
        가중치 초기화

        matrix: 대문자 변수명
        vertor: 소문자 변수명
        '''

        self.params = {}

        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        '''
        forward 연산

        x: 입력 데이터 (미니 배치로 학습할 경우 2차원 행렬로 들어옴)
        '''

        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1  # 입력층 => 은닉층
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2  # 은닉층 => 결과층
        y = softmax(a2)

        return y

    def loss(self, x, answer_label):
        '''
        x: 입력 데이터
        answer_label: 정답 레이블
        '''

        predict_label = self.predict(x)
        return mean_squared_error(predict_label, answer_label)

    def accuracy(self, x, answer_label):
        '''
        x: 입력 데이터
        answer_label: 정답 레이블
        '''

        y = self.predict(x)

        y = np.argmax(y, axis=1)
        answer_label = np.argmax(answer_label, axis=1)

        return np.sum(y == answer_label) / x.shape[0]

    def numerical_gradient(self, x, answer_label):
        '''
        x: 입력 데이터
        answer_label: 정답 레이블
        '''

        def loss(): return self.loss(x, answer_label)

        gradient = {}

        gradient['W1'] = numerical_gradient(loss, self.params['W1'])
        gradient['b1'] = numerical_gradient(loss, self.params['b1'])
        gradient['W2'] = numerical_gradient(loss, self.params['W2'])
        gradient['b2'] = numerical_gradient(loss, self.params['b2'])

        return gradient
