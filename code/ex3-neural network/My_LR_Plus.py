'''
    吴恩达老师的机器学习课程课后练习：逻辑回归实现多分类(手写体识别)
'''
#%%
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import classification_report
import matplotlib.cm

# 载入数据
def load_data(path, transpose=True):
    data = sio.loadmat(path)   # 载入 matlab 文件
    
    y = data.get('y')          # (5000, 1) 每行的一个元素都是一个向量
    y = y.reshape(y.shape[0])  # (5000, )  整体变为一个向量
    
    X = data.get('X')          # (5000, 400) 每行是20*20像素的图像每个点的灰度值

    if transpose:
        X = np.array([im.reshape(20, 20).T for im in X]) # 转置，得到正确的方向
        X = np.array([im.reshape(400) for im in X])      # 再次铺平成向量
    
    return X, y

raw_X, raw_y = load_data('ex3data1.mat')

X = np.insert(raw_X, 0, values=np.ones(raw_X.shape[0]), axis=1) # 插入第一列，全为1

# 由于matlab下标是从10开始： 10, 1, 2, 3 ...,所以将其转换成从0开始下标
y_matrix = []
for k in range(1, 11):
    y_matrix.append((raw_y==k).astype(int))   # 见配图 (向量化标签.png)

y_matrix = [y_matrix[-1]] + y_matrix[:-1]    # 将最后一列放到最前面
y = np.array(y_matrix)

# 训练一维模型
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def cost(theta, X, y):
    return np.mean(-y*np.log(sigmoid(X@theta)) - (1-y)*np.log(1-sigmoid(X@theta)))

def regularized_cost(theta, X, y, l=1):
    theta_1_n = theta[1:]
    regularized_term = (l/(2*len(X)))*np.power(theta_1_n, 2).sum()
    return cost(theta, X, y) + regularized_term

def gradient(theta, X, y):
    return (1/len(X))*X.T @ (sigmoid(X @ theta) - y)

def regularized_gradient(theta, X, y, l=1):
    theta_1_n = theta[1:]
    regularized_theta = (l/len(X))*theta_1_n
    regularized_term = np.concatenate([np.array([0]), regularized_theta])
    return gradient(theta, X, y) + regularized_term

def logisitci_regression(X, y, l=1):
    theta = np.zeros(X.shape[1])
    
    res = opt.minimize(fun = regularized_cost,
                       x0 = theta,
                       args = (X, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp':True})
    final_theta = res.x
    return final_theta

def predict(x, theta):
    prob = sigmoid(x @ theta)
    return (prob>=0.5).astype(int)

theta_0 = logisitci_regression(X, y[0])   # 从 向量化标签.png 中取出第一列作为标签训练(1 VS 9分类)

y_pred = predict(X, theta_0)
print('Accuracy={}'.format(np.mean(y_pred==y[0])))

# 训练K维模型
# 总共10中标签，每种标签进行一次 1 VS 9 分类训练
theta_k = np.array([logisitci_regression(X, y[k]) for k in range(10)])  #(10, 401)

# X @ theta.T = (5000, 401)*(10, 401).T = (5000, 10)
prob_matrix = sigmoid(X @ theta_k.T)
np.set_printoptions(suppress=True)

y_pred = np.argmax(prob_matrix, axis=1)   # 见配图：numpy_argmax.png

y_answer = raw_y.copy()
y_answer[y_answer==10] = 0
print(classification_report(y_answer, y_pred))














