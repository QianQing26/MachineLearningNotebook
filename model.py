import numpy as np
import matplotlib.pyplot as plt
import itertools
import GDoptimizer

class LinearRegression:
    '''线性回归'''
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
    '''生成多项式特征'''
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


class LogisticRegression:
    '''逻辑分类'''
    def __init__(self):
        self.w = None
        self.label =None
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def set_label(self,y):
        self.label = np.unique(y)

    def init_weights(self, num_features):
        limits = 1/np.sqrt(num_features)
        self.w = np.random.uniform(-limits,limits,(num_features+1,1))
        self.w[0]=0

    def fit(self, X, y, num_iteration=5000, batch_size = 16,optimizer=GDoptimizer.Adam()):
        num_samples, num_features = X.shape
        self.init_weights(num_features)

        self.set_label(y)
        y_hat = y
        for i in range(len(self.label)):
            y_hat[y_hat==self.label[i]]=i
        X_hat = np.insert(X, 0, 1, axis=1)
        y_hat = np.reshape(y_hat, (num_samples, 1))

        for _ in range(num_iteration):
            mask = np.random.choice(num_samples,batch_size)
            x_batch = X_hat[mask]
            y_batch = y_hat[mask]
            grad = self.grad(x_batch, y_batch)
            optimizer.update(self.w, 0.0,grad,0.0)

    def grad(self, X, y):
        h_x = X.dot(self.w)
        y_pred = self.sigmoid(h_x)
        return X.T.dot(y_pred-y)/X.shape[0]
    
    def predict(self, X):
        X_hat = np.insert(X,0,1,axis=1)
        y = self.sigmoid(-(X_hat.dot(self.w)))
        y = np.round(y)
        y.astype('int')
        y = np.reshape(y,(1,-1))[0]
        y[y==1.]=self.label[1]
        y[y==0.]=self.label[0]
        return y


class MultiLogisticRegresstion:
    '''使用softmax的多分类'''
    def __init__(self):
        self.w = None
        self.label = None
    
    def softmax(self,x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x = x - np.max(x)  # 溢出对策
        return np.exp(x) / np.sum(np.exp(x))

    def set_label(self,y):
        self.label = np.unique(y)

    def compute_prob(self,X):
        scores = X.dot(self.w)
        return self.softmax(scores)
    
    def compute_grad(self,X,y,regularization_param):
        num_samples = X.shape[0]
        probs = self.compute_prob(X)
        grad = -(1/num_samples)*X.T.dot(np.eye(self.w.shape[1])[y]-probs)
        regularization_term = regularization_param/num_samples*self.w
        grad+=regularization_term
        return grad 
    
    def initialize_weights(self,num_samples, num_features):
        limits = 1/np.sqrt(num_samples**2+num_features**2)
        self.w = np.random.uniform(-limits,limits,(num_features+1,num_samples))
    
    def fit(self,X,y,num_iterations=5000,regularization_param=0.01,batch_size=16,optimizer=GDoptimizer.SGD()):
        num_samples, num_features = X.shape
        X_hat = np.insert(X,0,1,axis=1)
        y_hat = y
        self.set_label(y_hat)
        for i in range(len(self.label)):
            y_hat[y_hat==self.label[i]]=i
        self.initialize_weights(num_samples, num_features)

        for i in range(num_iterations):
            mask = np.random.choice(num_samples,batch_size)
            x_batch = X_hat[mask]
            y_batch = y_hat[mask]
            grad = self.compute_grad(x_batch,y_batch,regularization_param)
            optimizer.update(self.w,0,grad,0)
    
    def predict(self,X):
        X_hat = np.insert(X,0,1,axis=1)
        y_pred = self.compute_prob(X_hat)
        return self.label[np.argmax(y_pred, axis=1)]



class KNN:
    def __init__(self, K=5):
        self.k = K
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def euclidean_distance(self, x1, x2):
        dis = x1-x2
        dis = dis*dis
        dis = np.sum(dis)
        return np.sqrt(dis)

    def predict(self,x):
        y_pred = np.zeros(x.shape[0])

        for i in range(x.shape[0]):
            distances = np.zeros((self.X.shape[0], 2))

            for j in range(self.X.shape[0]):
                dis = self.euclidean_distance(x[i], self.X[j])
                label = self.y[j]
                distances[j] = [dis, label]
            
            k_nearest_neighbors = distances[distances[:,1].argsort()][:self.k]
            counts = np.bincount(k_nearest_neighbors[:,1].astype('int'))
            testlabel = counts.argmax()
            y_pred[i] = testlabel
        
        return y_pred