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
