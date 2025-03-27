import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
DATA_NUM=60000
TEST_NUM=10000
history_loss=[]
class softmaxRegression:
    def __init__(self,learning_rate=0.01,iterations=200000):
        self.learning_rate=learning_rate
        self.iterations=iterations
        self.w=np.random.rand(784,10)
        self.b=np.random.rand(10)
    def z(self,k,x):
        return np.dot(x,self.w[:,k])+self.b[k]
    def softmax(self,x):
        z_set=np.array([self.z(k,x) for k in range(10)])
        return np.exp(z_set)/np.sum(np.exp(z_set))
    def loss(self,x,y):
        return -np.log(self.softmax(x)[y])
    def total_loss(self,X,Y):
        total_loss=0
        for i in range(DATA_NUM):
            for j in range(10):
                if j==Y[i]:
                    total_loss+=self.loss(X[i],Y[i])
        return total_loss/DATA_NUM
    def dw(self,x,y):
        return np.outer(x,self.softmax(x)-np.array([int(i==y) for i in range(10)]))
    def db(self,x,y):
        return self.softmax(x)-np.array([int(i==y) for i in range(10)])
    def sum_dw(self,X,Y):
        new_dw=np.zeros((784,10),dtype=float)
        for i in range(DATA_NUM):
            new_dw=new_dw+self.dw(X[i],Y[i])
        return new_dw/DATA_NUM
    def sum_db(self,X,Y):
        new_db=np.array([0.0 for _ in range(10)])
        for i in range(DATA_NUM):
            new_db=new_db+self.db(X[i],Y[i])
        return new_db/DATA_NUM
    def train(self,X,Y):
        global history_loss
        progress_bar = tqdm(range(self.iterations), desc="Training", unit="iter")
        for i in range(self.iterations):
            print(f"Iteration {i}")
            old_loss=self.total_loss(X,Y)
            history_loss.append((i,old_loss))
            d_w=self.sum_dw(X,Y)
            d_b=self.sum_db(X,Y)
            self.w=self.w-self.learning_rate*d_w
            self.b=self.b-self.learning_rate*d_b
            new_loss=self.total_loss(X,Y)
            progress_bar.set_postfix({"Loss": f"{new_loss:.4f}"})
            if old_loss-new_loss<0.0002:
                break
        progress_bar.close()
    def predict(self,x):
        max_label=-1
        min_posibility=-10
        posibility=self.softmax(x)
        for i in range(10):
            if posibility[i]>min_posibility:
                min_posibility=posibility[i]
                max_label=i
        return max_label
    def test(self,X_test,Y_test):
        count=0
        for i in range(TEST_NUM):
            if self.predict(X_test[i])==Y_test[i]:
                count+=1
        return count
    def draw_plot(self):
        global history_loss
        iterations=[item[0] for item in history_loss]
        losses=[item[1] for item in history_loss]
        plt.figure(figsize=(10,6))
        plt.plot(iterations,losses,'b-',linewidth=1,label='Training Losses')
        plt.xlabel('Iteration')
        plt.ylabel('Training Loss')
        plt.grid(True)
        plt.legend()
        plt.show()
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
X_train = train_images.reshape(-1, 784).astype(np.float32) / 255.0  # 形状 (60000, 784)
X_test = test_images.reshape(-1, 784).astype(np.float32) / 255.0   # 形状 (10000, 784)
y_train = train_labels.astype(np.int16)
y_test = test_labels.astype(np.int16)
softmax_regression=softmaxRegression()
softmax_regression.train(X_train,y_train)
count=softmax_regression.test(X_test,y_test)
print(f"Accurency: {count/TEST_NUM}")
softmax_regression.draw_plot()