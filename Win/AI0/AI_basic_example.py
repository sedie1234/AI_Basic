import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, pre_weight=0, pre_bias=0, learning_rate=0.01):
        self.param_weight = pre_weight
        self.param_bias = pre_bias
        self.learning_rate = learning_rate
    
    def inference(self, x):
        pred = self.param_weight * x + self.param_bias
        return pred

    def loss_func(self, y, pred):
        loss = np.mean((pred - y) ** 2)
        return loss
    
    def get_gradient(self, x, y, y_pred):
        gradient_weight = (2/len(x)) * np.sum((y_pred - y) * x)
        gradient_bias = (2/len(x)) * np.sum(y_pred - y)
        return gradient_weight, gradient_bias

    def train(self, datasets, epochs=1000):
        for epoch in range(epochs):
            y_pred = self.inference(datasets[0])

            loss = self.loss_func(datasets[1], y_pred)

            grad_w, grad_b = self.get_gradient(datasets[0], datasets[1], y_pred)

            self.param_weight -= self.learning_rate * grad_w
            self.param_bias -= self.learning_rate * grad_b

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}, Weight: {self.param_weight:.4f}, Bias: {self.param_bias:.4f}')

    def evaluateModel(self, datasets):
        pred = self.inference(datasets[0])

        eval = sum(pred) / sum(datasets[1]) * 100

        print(f"Model's Accuracy : {eval}%")

    def printParam(self):
        print(f"weight : {self.param_weight} / bias : {self.param_bias}")

    def printPlot(self, datasets):
        pred = self.inference(datasets[0])
        plt.scatter(datasets[0], datasets[1], color='blue', label='real_out')
        plt.plot(datasets[0], pred, color='red', label='pred_out')
        plt.title('real_out vs pred_out')
        plt.xlabel('input data')
        plt.ylabel('output data')
        plt.show()
    
# 학습 데이터 (x: 입력 데이터, y: 실제 출력(정답))
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.2, 4.4, 6.6, 8.8, 11.0])

# 평가 데이터 (test_x: 입력 데이터, test_y:실제 출력(정답))
test_x = np.array([6, 7, 8, 9, 10])
test_y = np.array([13.2, 15.4, 17.6, 19.8, 22.0])

datasets = [x, y]
test_datasets = [test_x, test_y]

learning_rate = 0.01

BasicNet = NeuralNetwork(learning_rate=learning_rate)

BasicNet.train(datasets, epochs=10)

BasicNet.evaluateModel(test_datasets)
BasicNet.printParam()
BasicNet.printPlot(test_datasets)

# for i in range(10):

#     BasicNet.train(datasets, epochs=10)

#     BasicNet.evaluateModel(test_datasets)
#     BasicNet.printParam()
#     BasicNet.printPlot(test_datasets)