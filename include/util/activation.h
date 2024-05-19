#pragma once

#include <functional>

enum ActivationType {
    RELU,
    SIGMOID,
    TANH
};

typedef std::function<double(double)> Activation;

Activation get_activation_function(ActivationType activation_function);

Activation get_activation_derivative_function(ActivationType activation_function);
