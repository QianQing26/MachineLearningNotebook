import numpy as np

class SGD:
    """Stochastic Gradient Descent"""
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self,w,b,dw,db):
        w -= self.lr*dw
        b -= self.lr*db
    

class Momentum:
    """Momenttum SGD"""
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, w,b,dw,db):
        if self.v is None:
            self.v = {}
            self.v['w'] = np.zeros_like(w)
            self.v['b'] = np.zeros_like(b)
        
        self.v['w'] = self.momentum*self.v['w'] - self.lr*dw
        self.v['b'] = self.momentum*self.v['b'] - self.lr*db

        w += self.v['w']
        b += self.v['b']


class Nesterov:
    """Nesterov's Accelerated Gradient"""
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, w, b, dw, db):
        if self.v is None:
            self.v = {}
            self.v['w'] = np.zeros_like(w)
            self.v['v'] = np.zeros_like(b)
        
        self.v['w'] *= self.momentum
        self.v['w'] -=self.lr * dw
        w += self.momentum * self.momentum * self.v['w']
        w -= (1+self.momentum) * self.lr * dw

        self.v['b'] *= self.momentum
        self.v['b'] -=self.lr * db
        b += self.momentum * self.momentum * self.v['b']
        b -= (1+self.momentum) * self.lr * db


class AdaGrad:
    """AdaGrad"""
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
    
    def update(self, w, b, dw, db):
        if self.h is None:
            self.h = {}
            self.h['w'] = np.zeros_like(w)
            self.h['b'] = np.zeros_like(b)
        
        self.h['w'] += dw * dw
        w -= self.lr*dw/(np.sqrt(self.h['w']) + 1e-7)
        self.h['b'] += db * db
        b -= self.lr*db/(np.sqrt(self.h['b']) + 1e-7)


class RMSprop:
    """PMSprop"""
    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, w, b, dw, db):
        if self.h is None:
            self.h = {}
            self.h['w'] = np.zeros_like(w)
            self.h['b'] = np.zeros_like(b)
        
        self.h['w'] *= self.decay_rate
        self.h['w'] += (1-self.decay_rate)*dw*dw
        w -= self.lr*self.h['w']/(np.sqrt(self.h['w'])+1e-7)

        self.h['b'] *= self.decay_rate
        self.h['b'] += (1-self.decay_rate)*db*db
        b -= self.lr*self.h['b']/(np.sqrt(self.h['b'])+1e-7)


class Adam:
    """Adam"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
    
    def update(self, w, b, dw, db):
        if self.m is None:
            self.m, self.v = {}, {}
            self.m['w'] = np.zeros_like(w)
            self.m['b'] = np.zeros_like(b)
            self.v['w'] = np.zeros_like(w)
            self.v['b'] = np.zeros_like(b)

        self.iter += 1
        lr_t = self.lr*np.sqrt(1.0-self.beta2**self.iter)/(1.0-self.beta1**self.iter)

        self.m['w'] += (1-self.beta1)*(dw-self.m['w'])
        self.v['w'] += (1-self.beta2)*(dw**2 - self.v['w'])
        w -= lr_t * self.m['w'] / (np.sqrt(self.v['w'])+1e-7)

        self.m['b'] += (1-self.beta1)*(db-self.m['b'])
        self.v['b'] += (1-self.beta2)*(db**2 - self.v['b'])
        b -= lr_t * self.m['b'] / (np.sqrt(self.v['b'])+1e-7)