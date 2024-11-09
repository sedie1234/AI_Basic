import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, pre_weight=0, pre_bias=0, learning_rate=0.01):
        # 가중치와 편향 초기화 및 학습률 설정
        self.param_weight = pre_weight
        self.param_bias = pre_bias
        self.learning_rate = learning_rate

        #epoch별 loss 기록
        self.losses = []
    
    def inference(self, x):
        # 선형 회귀 모델의 예측 값 계산 (y = wx + b)
        pred = self.param_weight * x + self.param_bias
        return pred

    def loss_func(self, y, pred):
        # 평균 제곱 오차 (MSE) 손실 함수 계산
        loss = np.mean((pred - y) ** 2)
        return loss
    
    def get_gradient(self, x, y, y_pred):
        # 가중치와 편향에 대한 기울기 계산
        gradient_weight = (2/len(x)) * np.sum((y_pred - y) * x)
        gradient_bias = (2/len(x)) * np.sum(y_pred - y)
        return gradient_weight, gradient_bias

    def train(self, datasets, epochs=1000):
        # 주어진 epoch 동안 모델 학습
        for epoch in range(epochs):
            # 현재 가중치와 편향을 사용하여 예측값 계산
            y_pred = self.inference(datasets[0])

            # 현재 예측값과 실제 값의 손실 계산
            loss = self.loss_func(datasets[1], y_pred)

            # epoch별 loss 기록
            self.losses.append(loss)

            # 가중치와 편향의 기울기 계산
            grad_w, grad_b = self.get_gradient(datasets[0], datasets[1], y_pred)

            # 가중치와 편향 업데이트 (경사 하강법 적용)
            self.param_weight -= self.learning_rate * grad_w
            self.param_bias -= self.learning_rate * grad_b

            # 100 epoch마다 손실 및 파라미터 출력
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}, Weight: {self.param_weight:.4f}, Bias: {self.param_bias:.4f}')

    def evaluateModel(self, datasets):
        # 주어진 데이터셋에 대해 모델의 성능 평가
        pred = self.inference(datasets[0])

        # 예측 값과 실제 값의 합을 비교하여 모델의 성능을 퍼센트로 표현
        eval = sum(pred) / sum(datasets[1]) * 100

        print(f"Model's Accuracy : {eval}%")

    def printParam(self):
        # 현재 가중치와 편향 출력
        print(f"weight : {self.param_weight} / bias : {self.param_bias}")

    def printPlot(self, datasets):
        # 실제 출력과 예측 출력을 비교하는 플롯 생성
        pred = self.inference(datasets[0])
        plt.scatter(datasets[0], datasets[1], color='blue', label='real_out')  # 실제 데이터 시각화
        plt.plot(datasets[0], pred, color='red', label='pred_out')  # 예측 데이터 시각화
        plt.title('real_out vs pred_out')
        plt.xlabel('input data')
        plt.ylabel('output data')
        plt.legend()
        plt.show()
    
    def printLoss(self):
        # epoch별 loss 출력
        plt.plot(self.losses, label='Loss over epochs')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

# 학습 데이터 (x: 입력 데이터, y: 실제 출력(정답))
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.2, 4.4, 6.6, 8.8, 11.0])

# 평가 데이터 (test_x: 입력 데이터, test_y: 실제 출력(정답))
test_x = np.array([6, 7, 8, 9, 10])
test_y = np.array([13.2, 15.4, 17.6, 19.8, 22.0])

# 데이터셋 패킹
datasets = [x, y]
test_datasets = [test_x, test_y]

# 학습률 설정
learning_rate = 0.01

# NeuralNetwork 인스턴스 생성 및 학습
BasicNet = NeuralNetwork(learning_rate=learning_rate)

# 학습 수행
BasicNet.train(datasets, epochs=10)

# 모델 평가 및 파라미터 출력
BasicNet.evaluateModel(test_datasets)
BasicNet.printParam()

# 모델 결과 플롯
BasicNet.printPlot(test_datasets)

# 학습에 따른 Loss
# BasicNet.printLoss()

# 추가 학습 루프 (epoch에 따라 Accuracy의 변화를 관찰)
# for i in range(10):
#     BasicNet.train(datasets, epochs=10)
#     BasicNet.evaluateModel(test_datasets)
#     BasicNet.printParam()
#     BasicNet.printPlot(test_datasets)
