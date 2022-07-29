import time
import numpy as np
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras import optimizers
from scipy.io import savemat
import matplotlib as mpl
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']
mpl.rcParams['axes.unicode_minus'] = False


### 模型

# 划分数据集
def split_dataset(data):
    # 总数据为四年用气情况，前三年为训练集，最后一年为测试集。
    train, test = data[1:-328], data[-328:-6]
    # 预测目标为未来一周的用气情况，所以价将数据以周为单位划分。
    train = array(split(train, len(train) / 7))
    test = array(split(test, len(test) / 7))
    return train, test


# 计算预测值的RMSE
def evaluate_forecasts(actual, predicted):
    scores = list()
    # 计算每一天预测值的RMSE
    for i in range(actual.shape[1]):
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        rmse = sqrt(mse)
        scores.append(rmse)
    # 计算总RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


# 将预处理的数据转为适用于LSTM的输入和输出格式
def to_supervised(train, n_input, n_out=7):
    # 数据扁平化
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # 将输入输出数据对应
    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_out
        # 输入输出单位内数据是完整的，抹掉末尾不整齐的数据
        if out_end <= len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        in_start += 1
    return array(X), array(y)


# 训练模型
def build_model(train, h1, h2, time_step, learning_rate, max_iter, batch_size):
    # 数据预处理
    train_x, train_y = to_supervised(train, time_step)
    # 定义模型参数
    verbose, epochs, batch_size = 0, max_iter, batch_size
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # LSTM输入格式 [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    # 定义函数
    model = Sequential()
    model.add(LSTM(h1, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(h2, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    # 学习率采用指数下降方案
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=max_iter,
        decay_rate=0.9
    )
    optimizer = optimizers.Adam(learning_rate=lr_schedule)
    model.compile(loss='mse', optimizer=optimizer)
    model.compile(loss='mse', optimizer='adam')
    # 网络训练
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


# 模型预测
def forecast(model, history, n_input):
    # 格式对其
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # 训练集的最后一个单位数据为测试集的第一个单位数据
    input_x = data[-n_input:, 0]
    input_x = input_x.reshape((1, len(input_x), 1))
    # 预测下一周的数值
    yhat = model.predict(input_x, verbose=0)
    yhat = yhat[0]
    return yhat


# 评价模型
def evaluate_model(train, test, h1, h2, time_step, learning_rate, max_iter, batch_size):
    # 定义模型并训练
    model = build_model(train, h1, h2, time_step, learning_rate, max_iter, batch_size)
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        # 预测下周的数据
        yhat_sequence = forecast(model, history, time_step)
        predictions.append(yhat_sequence)
        # 真数数值加入历史预测下一周
        history.append(test[i, :])
    # 模型预测
    predictions = array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores

# CPSO
# 重新分配边界外的参数，在边界内的生成一个随机数
def boundary(pop):
    # 防止粒子跳出范围
    pop[0] = int(pop[0])
    pop[1] = int(pop[1])
    pop[2] = int(pop[2])
    pop[4] = int(pop[4])  # 迭代数和节点数都应为整数
    pop[5] = int(pop[5])
    if pop[0] > 200 or pop[0] < 1:
        pop[0] = np.random.randint(1, 200)
    if pop[1] > 200 or pop[1] < 1:
        pop[1] = np.random.randint(1, 200)
    if pop[2] > 31 or pop[2] < 1:
        pop[2] = np.random.randint(1, 14)
    if pop[3] > 0.1 or pop[3] < 0.001:
        pop[3] = (0.01 - 0.001) * np.random.rand() + 0.001
    if pop[4] > 500 or pop[4] < 1:
        pop[4] = np.random.randint(1, 500)
    if pop[5] > 128 or pop[5] < 1:
        pop[5] = np.random.randint(1, 128)
    return pop


# 生成混沌序列
def logistic_map(number, u=4):
    log_set = np.zeros(number)
    log_set[0] = np.random.uniform()

    for i in np.arange(1, number):
        log_set[i] = u * log_set[i - 1] * (1 - log_set[i - 1])

    return log_set


def PSO(pN, fes):
    # PSO参数设置
    dim = 6
    pN = pN
    fes = fes
    fes_iteration = 0
    w = 0.8; c1 = 2.0; c2 = 2.0;
    # 初始化
    X = np.zeros((pN, dim))
    V = np.zeros((pN, dim))
    pbest = np.zeros((pN, dim))
    gbest = np.zeros((1, dim))
    p_fit = np.zeros(pN)
    result = []
    fit = 10000
    for j in range(dim):
        # logistic混沌映射，针对整个粒子群来看，每一个维度都是均匀混沌的
        log_set = logistic_map(pN)
        for i in range(pN):
            if j == 0:
                X[i][j] = 1 + int(log_set[i] * (200 - 1))
            elif j == 1:
                X[i][j] = 1 + int(log_set[i] * (200 - 1))
            elif j == 2:
                X[i][j] = 1 + int(log_set[i] * (14 - 1))
            elif j == 3:
                X[i][j] = 0.001 + (log_set[i] * (0.01 - 0.001))
            elif j == 4:
                X[i][j] = 1 + int(log_set[i] * (500 - 1))
            elif j == 5:
                X[i][j] = 1 + int(log_set[i] * (128 - 1))
            V[i][j] = np.random.rand()
    for i in range(pN):
        pbest[i] = X[i]
        pop = list(X[i, :])
        tmp = float(fitness(pop))
        p_fit[i] = tmp
        if (tmp < fit):
            fit = tmp
            gbest = X[i]
            # 开始循环迭代
    trace = []
    history_fitness_list = []
    history_position_list = []
    while fes_iteration < fes :
        print()
        print("epoch %s ：" %(fes_iteration), end=" ")
        for i in range(pN):  # 更新gbest\pbest
            pop = list(X[i, :])
            temp = float(fitness(pop))
            fes_iteration = fes_iteration + 1

            history_position_list.append(X[i, :])
            history_fitness_list.append(temp)
            print(str(round(temp, 3)), end=" ")

            if (temp < p_fit[i]):  # 更新个体最优
                p_fit[i] = temp
                pbest[i, :] = X[i, :]
                if (p_fit[i] < fit):  # 更新全局最优
                    gbest = X[i, :].copy()
                    fit = p_fit[i].copy()
            result.append(gbest)
            trace.append(fit)
        for i in range(pN):
            r1, r2 = np.random.random(), np.random.random()
            V[i, :] = w * V[i, :] + c1 * r1 * (pbest[i] - X[i, :]) + c2 * r2 * (gbest - X[i, :])
            X[i, :] = X[i, :] + V[i, :]
            X[i, :] = boundary(X[i, :])  # 边界判断
    return trace, result, history_fitness_list, history_position_list


def fitness(pop):
    h1 = int(pop[0])
    h2 = int(pop[1])
    time_step = int(pop[2])
    learning_rate = pop[3]
    max_iter = int(pop[4])
    batch_size = int(pop[5])
    score, scores = evaluate_model(train, test, h1, h2, time_step, learning_rate, max_iter, batch_size)
#     score = sum(pop)
    return score

# 数据加载
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# 数据划分
train, test = split_dataset(dataset.values)

pN = 10
fes = 500

trace, result, history_fitness_list, history_position_list = PSO(pN, fes)

time_flag = str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
fitness_flag = "fitness" + str(round(trace[-1], 6)) + "_"
data_flag = fitness_flag + time_flag + ".mat"

savemat(data_flag, {'fitness': trace, 'gbest': result,
                    'history_fitness': history_fitness_list,
                    'history_position': history_position_list})

fitness_image = fitness_flag + time_flag + ".png"
plt.figure()
plt.plot(trace)
plt.title('适应度值收敛图')
plt.xlabel('运行代数')
plt.ylabel('适应度值')
plt.savefig(fitness_image)