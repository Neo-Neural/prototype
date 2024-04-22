#include "model/ann.h"

#include "util/activation.h"

#include <random>
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
    std::random_device rd;
    std::mt19937 gen(rd());
    for (auto i = 0; i <= layers_num; i++) {
        int fout = i == layers_num ? output_dim : layers[i];
        int fin = i == 0 ? input_dim : layers[i - 1];
        std::normal_distribution<> dis(0.0, sqrt(2.0 / (fin + fout)));
        arma::mat weight(fout, fin);
        weight.transform([&](double x) {
            return dis(gen);
        });

        this->weights.push_back(weight);
    }
}

double ANN::single_test(const TestPoint &test_point) {
    arma::vec z(test_point.input);
    for (auto i = 0; i < this->layers_num; i ++) {
        z = this->weights[i] * z + this->biases[i];
        z.transform(Activation::tanh);
    }
    z = this->weights[this->layers_num] * z - test_point.answer;
    z.transform(Activation::tanh);

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