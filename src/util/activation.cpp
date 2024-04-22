#include <cmath>
#include "util/activation.h"

namespace Activation {
    double relu(double x) {
        return x > 0 ? x : 0;
    }

    double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }

    double tanh(double x) {
        return 2 / (1 + exp(-2 * x)) - 1;
    }
};

namespace ActivationDifferential {
    double relu(double x) {
        return x > 0 ? 1 : 0;
    }

    double sigmoid(double x) {
        double sig = Activation::sigmoid(x);
        return sig * (1 - sig);
    }

    double tanh(double x) {
        double tanh = Activation::tanh(x);
        return 1 - tanh * tanh;
    }
};
