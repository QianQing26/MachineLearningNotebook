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


import numpy as np

class KMeans:
    # 调用train方法进行训练
    def __init__(self, data, num_clusters):
        self.data = data
        self.num_clusters = num_clusters

    def train(self, num_iters):
        # 1.随机选择质心
        centroids = self.centroids_init(self.data, self.num_clusters)
        # 2.开始训练
        num_samples = self.data.shape[0]
        closest_centroids_ids = np.empty((num_samples,1))
        for _ in range(num_iters):
            # 3.样本点归属划分
            closest_centroids_ids = self.find_closest(self.data, centroids)
            # 4.中心点迭代过程
            centroids = self.centroids_compute(self.data, closest_centroids_ids, self.num_clusters)
        return centroids,closest_centroids_ids
    
    def centroids_init(self,data, num_clusters):
        num_samples = data.shape[0]
        randomID = np.random.permutation(num_samples)
        cenrtroids = data[randomID[:num_clusters],:]
        return cenrtroids

    def find_closest(self,data,centroids):
        closest_ids = []
        for sample in data:
            distances = []
            for cen in centroids:
                dis = sample - cen
                dis = dis*dis
                dis = np.sum(dis)
                dis = np.sqrt(dis)
                distances.append(dis)
            closest_ids.append(distances.index(min(distances)))
        closest_centroids_ids = np.array(closest_ids)
        return closest_centroids_ids
    
    def centroids_compute(self,data,closest_centroids_ids, num_clusters):
        # 4,5'
        num_features = data.shape[1]
        centroids = np.zeros((num_clusters,num_features))
        for centroids_id in range(num_clusters):
            points = data[closest_centroids_ids==centroids_id]
            centroids[centroids_id] = np.mean(points,axis=0)
        return centroids
    

class DBSCAN:
    # 调用fit方法进行训练
    def __init__(self, eps, min_samples):
        self.eps = eps                      # 邻域半径参数
        self.min_samples = min_samples      # 密度参数
        self.labels = None                  
    
    def distance(self, xa, xb):
        # 计算两点的欧氏距离
        return np.sqrt(np.sum((xa - xb) ** 2))
    
    def region_query(self, X, point_idx):
        # 找到邻域内的点的索引
        neighbors = []
        for i in range(len(X)):
            if self.distance(X[point_idx], X[i]) <= self.eps:
                neighbors.append(i)
        return neighbors
    
    def expand_cluster(self, X, cluster_id, point_idx, neighbors):
        # 根据给定点及其邻域进行拓展
        self.labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            if self.labels[neighbor] == -1:
                self.labels[neighbor] = cluster_id
            elif self.labels[neighbor] == 0:
                self.labels[neighbor] = cluster_id
                point_neighbors = self.region_query(X, neighbor)
                if len(point_neighbors) >= self.min_samples:
                    neighbors += point_neighbors
            i += 1
    
    def fit(self, X):
        # 训练
        n = len(X)
        self.labels = np.zeros(n)
        cluster_id = 1
        for i in range(n):
            if self.labels[i] != 0:
                continue
            neighbors = self.region_query(X, i)
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1
            else:
                self.expand_cluster(X, cluster_id, i, neighbors)
                cluster_id += 1
        # 返回聚类标签
        return self.labels
    


class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, 
                 value=None, left=None, right=None):
        self.feature_index = feature_index   # 特征索引
        self.threshold = threshold           # 分割阈值
        self.value = value                   # 结点值
        self.left = left                     # 左子结点
        self.right = right                   # 右子结点

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def _gini_index(self, y):
        classes = np.unique(y)
        gini = 0.0
        for cls in classes:
            p = np.sum(y == cls) / len(y)
            gini += p * (1 - p)
        return gini
    
    def _best_split(self, X, y):
        best_gini = float('inf')
        best_feature_index = 0
        best_threshold = 0
        for feature_index in range(X.shape[1]):
            for threshold in np.unique(X[:, feature_index]):
                left_mask = X[:, feature_index] <= threshold
                right_mask = X[:, feature_index] > threshold
                gini = (len(y[left_mask]) * self._gini_index(y[left_mask])
                        + len(y[right_mask]) * self._gini_index(y[right_mask])) / len(y)
                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold
        return best_feature_index, best_threshold
    
    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return DecisionNode(value=np.argmax(np.bincount(y)))
        feature_index, threshold = self._best_split(X, y)
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return DecisionNode(feature_index, threshold, left=left, right=right)
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, 0)
        
    def predict(self, X):
        results = []
        for sample in X:
            node = self.tree
            while node.left:
                if sample[node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            results.append(node.value)
        return results