import time
import numpy
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


# 定义个体类
class Individual:
    def __init__(self, params, fitness):
        self.params = params
        self.fitness = fitness

    def __repr__(self):
        return 'params: {} fitness: {}'.format(self.params, self.fitness)


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


# 重新分配边界外的参数，在边界内的生成一个随机数
def bounded(params, min_s: list, max_s: list):
    return numpy.array([numpy.random.uniform(min_s[d], max_s[d])
                        if params[d] < min_s[d] or params[d] > max_s[d]
                        else params[d]
                        for d in range(len(params))])


# 个体初始化
def generate_individual(benchmark, min_s, max_s, dimensions):
    params = numpy.random.uniform(min_s, max_s, dimensions)
    fitness = benchmark(params)
    return Individual(params, fitness)


# 种群初始化
def generate_population(size, min_s, max_s, dimensions, benchmark):
    return [generate_individual(benchmark, min_s, max_s, dimensions) for _ in range(size)]


def generate_prt_vector(prt, dimensions):
    return numpy.random.choice([0, 1], dimensions, p=[prt, 1 - prt])


# 根据适应度找到种群的leader
def get_leader(population):
    return min(population, key=lambda individual: individual.fitness)


# SOMA
def soma_all_to_one(population, prt, path_length, step, fes, min_s, max_s, dimensions, benchmark):
    fes_iteration = 0
    history_fitness_list = []
    history_position_list = []
    history_min_fitness_list = []
    global_min_fitness = numpy.Inf

    while fes_iteration < fes:  # check fes
        print()
        leader = get_leader(population)

        for index, individual in enumerate(population):
            if fes_iteration >= fes:  # check fes
                break

            if individual is leader:
                print()
                print("population %s is leader" % (index), end=" ")
                continue
            print()
            print("population %s :" % (index), end=" ")

            next_position = individual.params
            prt_vector = generate_prt_vector(prt, dimensions)

            for t in numpy.arange(step, path_length, step):
                if fes_iteration >= fes:  # check fes
                    break

                current_position = individual.params + (leader.params - individual.params) * t * prt_vector
                current_position = bounded(current_position, min_s, max_s)
                fitness = benchmark(current_position)

                history_fitness_list.append(fitness)
                history_position_list.append(current_position)
                print(str(round(fitness, 3)), end=" ")


                fes_iteration += 1  # increment fes iteration

                if fitness <= individual.fitness:
                    next_position = current_position
                    individual.fitness = fitness

                if fitness < global_min_fitness:
                    global_min_fitness = fitness

                history_min_fitness_list.append(global_min_fitness)

            individual.params = next_position

    return get_leader(population), history_fitness_list, history_position_list, history_min_fitness_list


def run_algorithm(dimensions, min_border, max_border, benchmark):
    fes = 500
    min_s = min_border
    max_s = max_border

    population = generate_population(pop_size, min_s, max_s, dimensions, benchmark)
    winner, history_fitness_list, history_position_list, history_min_fitness_list = soma_all_to_one \
        (population, prt, path_lenght, step, fes, min_s, max_s, dimensions, benchmark)
    print()
    print(f"fitness: {winner.fitness}")

    return winner.fitness, history_fitness_list, history_position_list, history_min_fitness_list


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


# SOMA 参数
pop_size = 10
prt = 0.1
path_lenght = 2
step = 0.21

# 加载数据
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True,
                   parse_dates=['datetime'], index_col=['datetime'])
# 数据划分
train, test = split_dataset(dataset.values)
winner, history_fitness_list, history_position_list, history_min_fitness_list = run_algorithm(6, [1, 1, 1, 0.001, 1, 1], [200, 200, 14, 0.1, 500, 128], fitness)


time_flag = str(time.strftime("%m-%d_%H:%M:%S", time.localtime()))
fitness_flag = "fitness:" + str(round(winner, 3)) + "_"

data_flag = fitness_flag + time_flag + ".mat"
savemat(data_flag, {'history_fitness': history_fitness_list,
                    'history_position': history_position_list,
                    'history_min_fitness': history_min_fitness_list})

fitness_image = fitness_flag + time_flag + ".png"
plt.figure()
plt.plot(history_min_fitness_list)
plt.title('适应度值收敛图')
plt.xlabel('运行代数')
plt.ylabel('适应度值')
plt.savefig(fitness_image)