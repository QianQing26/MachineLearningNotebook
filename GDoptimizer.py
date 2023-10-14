import numpy as np

class SGD:
    """Stochastic Gradient Descent"""
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self,w,b,dw,db):
        w -=self.lr*dw
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


class AdaMax:
    """AdaMax"""
    def __init__(self, lr=0.002, beta1=0.9, beta2=0.999):
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
        lr_t = self.lr / (1.0 - self.beta1**self.iter)
        self.m['w'] = self.beta1*self.m['w'] + (1-self.beta1)*dw
        self.v['w'] = np.maximum(self.beta2*self.v['w'], np.abs(dw))
        w -= lr_t * self.m['w'] / (self.v['w'] + 1e-7)
        self.m['b'] = self.beta1*self.m['b'] + (1-self.beta1)*db
        self.v['b'] = np.maximum(self.beta2*self.v['b'], np.abs(db))
        b -= lr_t * self.m['b'] / (self.v['b'] + 1e-7)


class Adadelta:
    def __init__(self, rho=0.95, epsilon=1e-6):
        self.rho = rho
        self.epsilon = epsilon
        self.E_dw = None
        self.E_db = None
        self.E_dw_update = None
        self.E_db_update = None
    
    def update(self, w, b, dw, db):
        if self.E_dw is None:
            self.E_dw = np.zeros_like(w)
            self.E_db = np.zeros_like(b)
            self.E_dw_update = np.zeros_like(w)
            self.E_db_update = np.zeros_like(b)
        
        self.E_dw = self.rho*self.E_dw + (1-self.rho)*dw**2
        self.E_db = self.rho*self.E_db + (1-self.rho)*db**2
        dw_update = -np.sqrt(self.E_dw_update + self.epsilon)/np.sqrt(self.E_dw + self.epsilon)*dw
        db_update = -np.sqrt(self.E_db_update + self.epsilon)/np.sqrt(self.E_db + self.epsilon)*db
        w += dw_update
        b += db_update
        self.E_dw_update = self.rho*self.E_dw_update + (1-self.rho)*dw_update**2
        self.E_db_update = self.rho*self.E_db_update + (1-self.rho)*db_update**2


class Adafactor:
    def __init__(self, learning_rate=None, beta1=0.9, beta2=0.999, epsilon=1e-8, regularization=None, decay_rate=-0.8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.regularization = regularization
        self.decay_rate = decay_rate
        self.m = None
        self.v = None
        self.t = None
    
    def update(self, w, b, dw, db):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
            self.t = 0
        
        self.t += 1
        self.m = self.beta1*self.m + (1-self.beta1)*dw
        self.v = self.beta2*self.v + (1-self.beta2)*(dw**2)
        lr = self.learning_rate
        if self.learning_rate is None:
            lr = (self.learning_rate/np.sqrt(self.t))
        
        if self.regularization == "l1":
            w -= lr * (self.m/np.sqrt(self.v + self.epsilon) + self.decay_rate * np.sign(w))
        elif self.regularization == "l2":
            w -= lr * (self.m/np.sqrt(self.v + self.epsilon) + self.decay_rate * w)
        else:
            w -= lr * (self.m/np.sqrt(self.v + self.epsilon))
        self.m = self.beta1*self.m + (1-self.beta1)*db
        self.v = self.beta2*self.v + (1-self.beta2)*(db**2)
        lr = self.learning_rate
        if self.learning_rate is None:
            lr = (self.learning_rate/np.sqrt(self.t))
        
        if self.regularization == "l1":
            b -= lr * (self.m/np.sqrt(self.v + self.epsilon) + self.decay_rate * np.sign(b))
        elif self.regularization == "l2":
            b -= lr * (self.m/np.sqrt(self.v + self.epsilon) + self.decay_rate * b)
        else:
            b -= lr * (self.m/np.sqrt(self.v + self.epsilon))


class AMSGrad:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.v_hat = None
    
    def update(self, w, b, dw, db):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
            self.v_hat = np.zeros_like(w)
        
        self.m = self.beta1*self.m + (1-self.beta1)*dw
        self.v = self.beta2*self.v + (1-self.beta2)*(dw**2)
        self.v_hat = np.maximum(self.v_hat, self.v)
        w -= self.learning_rate * (self.m/(np.sqrt(self.v_hat) + self.epsilon))
        self.m = self.beta1*self.m + (1-self.beta1)*db
        self.v = self.beta2*self.v + (1-self.beta2)*(db**2)
        self.v_hat = np.maximum(self.v_hat, self.v)
        b -= self.learning_rate * (self.m/(np.sqrt(self.v_hat) + self.epsilon))


class RAdam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=-0.8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.iter = 0
        self.rho_inf = 2/(1-self.beta2) - 1
        self.rho_t = None
        self.m = None
        self.v = None
    
    def update(self, w, b, dw, db):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
        
        self.iter += 1
        rho_t = self.rho_inf - 2*self.iter*self.beta2**self.iter / (1-self.beta2**self.iter)
        self.m = self.beta1*self.m + (1-self.beta1)*dw
        self.v = self.beta2*self.v + (1-self.beta2)*(dw**2)
        if rho_t > 4:
            r = np.sqrt(((rho_t-4)*(rho_t-2)*self.rho_inf)/((self.rho_inf-4)*(self.rho_inf-2)*rho_t))
            w -= self.learning_rate * r * (self.m/(np.sqrt(self.v) + self.epsilon) + self.decay_rate * w)
        else:
            w -= self.learning_rate * (self.m/(np.sqrt(self.v) + self.epsilon) + self.decay_rate * w)
        self.m = self.beta1*self.m + (1-self.beta1)*db
        self.v = self.beta2*self.v + (1-self.beta2)*(db**2)
        if rho_t > 4:
            r = np.sqrt(((rho_t-4)*(rho_t-2)*self.rho_inf)/((self.rho_inf-4)*(self.rho_inf-2)*rho_t))
            b -= self.learning_rate * r * (self.m/(np.sqrt(self.v) + self.epsilon) + self.decay_rate * b)
        else:
            b -= self.learning_rate * (self.m/(np.sqrt(self.v) + self.epsilon) + self.decay_rate * b)

'''
L-BFGS
Proximal Gradient Descent
'''