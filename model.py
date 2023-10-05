import numpy as np
import matplotlib.pyplot as plt
import itertools
import GDoptimizer

class LinearRegression:
    def __init__(self,learning_rate=0.01,num_iterations=50,batch_size=16):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size=batch_size
        self.weights=None
        self.bias=None

    def fit(self, X, y,optimizer=GDoptimizer.SGD()):
        loss_his = []
        num_samples, num_features = X.shape
        self.weights=np.random.rand(num_features)
        self.bias = np.random.rand()
        for iter in range(self.num_iterations):
            # 抽取batch
            mask = np.random.choice(num_samples,self.batch_size)
            batch_X=X[mask]
            batch_y=y[mask]

            dw, db = self.gradient(batch_X,batch_y)

            # dw = np.dot(batch_X.T,(y_pred-batch_y))/self.batch_size
            # db = np.sum(y_pred-batch_y)/self.batch_size

            # self.weights -= self.learning_rate*dw
            # self.bias-=self.learning_rate*db
            optimizer.update(self.weights, self.bias, dw, db)

            if iter%10 ==0:
                loss_his.append(self.loss(X,y))
        
        # x = [i for i in range(1,len(loss_his)+1)]
        x = np.arange(1,len(loss_his)+1)
        loss_his = np.array(loss_his)
        plt.plot(x,loss_his,label='training_loss')
        plt.legend()
        plt.title('training_loss')
        plt.show()

    def predict(self,X):
        return np.dot(X,self.weights)+self.bias
    
    def loss(self,data,label):
        pred = self.predict(data)
        # print(pred.shape)
        # print(label.shape)
        delta = pred-label 
        loss = 0.5*np.dot(delta.T,delta)
        return loss

    def gradient(self,x,t):
        y = self.predict(x)
        dw = np.dot(x.T,(y-t))/self.batch_size
        db = np.sum(y-t)/self.batch_size
        return dw,db



class PolynomialFeatures:
    def __init__(self, degree=2):
        self.degree = degree
    
    def fit_transform(self, X):
        
        n_samples, n_features = X.shape
        # print(X.shape)
        X_poly = np.ones((n_samples, 1))
        
        for d in range(1, self.degree + 1):
            combinations = itertools.combinations_with_replacement(range(n_features), d)

            for comb in combinations:
                new_feature = np.prod(X[:, comb], axis=1).reshape(-1, 1)
                X_poly = np.hstack((X_poly, new_feature))
        
        return X_poly[:, 1:]