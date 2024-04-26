import json
import numpy as np
from matplotlib import pyplot as plt

def propagation(layers_num, weight, bias, input):
    for i in range(layers_num):
        input = np.dot(weight[i], input) + bias[i]
        input = np.tanh(input)
    result = np.tanh(np.dot(weight[-1], input))
    
    # find the max elem of result
    return np.argmax(result)

def unwrap(lst):
    return [i[0] for i in lst], [i[1] for i in lst]

if __name__ == "__main__":

    filename = input("请输入文件名：")
    f = open(filename, encoding="utf-8")

    data = json.load(f)

    layers_num = data["layers_num"]
    layers_sizes = [2] + data["layers_sizes"] + [4]
    print(layers_sizes)
    
    bias = [np.array(i) for i in data["biases"]]
    weight = [np.array(i) for i in data["weights"]]
    
    for i in range(layers_num + 1):
        weight[i] = weight[i].reshape(layers_sizes[i], layers_sizes[i+1]).transpose()
    
    output = [[], [], [], []]
    for x in np.linspace(-1.0, 1.0, 100):
        for y in np.linspace(-1.0, 1.0, 100):
            input = np.array([x, y])
            result = propagation(layers_num, weight, bias, input)
            output[result].append(input)
    
    plt.scatter(*unwrap(output[0]), color="r")
    plt.scatter(*unwrap(output[1]), color="g")
    plt.scatter(*unwrap(output[2]), color="b")
    plt.scatter(*unwrap(output[3]), color="y")
    
    plt.show()
    