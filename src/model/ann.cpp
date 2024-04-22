#include "model/ann.h"

#include "util/activation.h"
using namespace Activation;

ANN::ANN(const int input_dim, const std::vector<int> &layers, const int output_dim) {
    this->layers_num = layers.size();
    this->layers_sizes.assign(layers.begin(), layers.end());
    
    // Init biases
    for (auto i = 0; i < layers_num; i++) {
        arma::vec bias(layers[i], arma::fill::zeros);
        this->biases.push_back(bias);
    }

    // Init weights
    for (auto i = 0; i <= layers_num; i++) {
        arma::mat weight(
            i == layers_num ? output_dim : layers[i],
            i == 0 ? input_dim : layers[i - 1],
            arma::fill::randu
        );
        this->weights.push_back(weight);
    }
}

double ANN::single_test(const TestPoint &test_point) {
    arma::vec z(test_point.input);
    for (auto i = 0; i < this->layers_num; i ++) {
        z = z * this->weights[i] + this->biases[i];
        z.transform(Activation::tanh);
    }
    z = z * this->weights[this->layers_num] - test_point.answer;
    double loss = 0.0;
    z.for_each([&loss](double x) {
        loss += x * x;
    });
    return loss;
}

void ANN::train(const Data &data) {
    
}

void ANN::save(const std::string &path) {
    
}

void ANN::load(const std::string &path) {
    
}