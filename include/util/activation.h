#pragma once

namespace Activation {
    double relu(double x);
    double sigmoid(double x);
    double tanh(double x);
}

namespace ActivationDifferential {
    double relu(double x);
    double sigmoid(double x);
    double tanh(double x);
};
