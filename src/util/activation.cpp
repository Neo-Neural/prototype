#include <cmath>
#include "util/activation.h"

double relu(double x) {
    return x > 0 ? x : 0;
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double tanh_(double x) {
    return 2 / (1 + exp(-2 * x)) - 1;
}

double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

double sigmoid_derivative(double x) {
    double sig = sigmoid(x);
    return sig * (1 - sig);
}

double tanh_derivative(double x) {
    double tanh = tanh_(x);
    return 1 - tanh * tanh;
}

Activation get_activation_function(ActivationType activation_function) {
    switch (activation_function) {
        case ActivationType::RELU:
            return relu;
        case ActivationType::SIGMOID:
            return sigmoid;
        case ActivationType::TANH:
            return tanh_;
        default:
            // will never reach
            break;
    }
}

Activation get_activation_derivative_function(ActivationType activation_function) {
    switch (activation_function) {
        case ActivationType::RELU:
            return relu_derivative;
        case ActivationType::SIGMOID:
            return sigmoid_derivative;
        case ActivationType::TANH:
            return tanh_derivative;
        default:
            // will never reach
            break;
    }
}
