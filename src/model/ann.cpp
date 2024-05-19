#include "model/ann.h"

#include "data/data.h"
#include "util/activation.h"

#include <random>
#include <vector>
#include <nlohmann/json.hpp>

ANN::ANN(const std::string &path) {
    this->load(path);
}

ANN::ANN(const int input_dim, const std::vector<int> &layers, const int output_dim, ActivationType activate) {
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
        int fin = i == 0 ? input_dim : layers[i - 1];
        int fout = i == layers_num ? output_dim : layers[i];
        std::normal_distribution<> dis(0.0, sqrt(2.0 / (fin + fout)));
        arma::mat weight(fout, fin);
        weight.transform([&](double x) {
            return dis(gen);
        });

        this->weights.push_back(weight);
    }

    this->activate = activate;
}

ANN::~ANN() {}

double ANN::single_test(const TestPoint &test_point) {
    arma::vec z(test_point.input);

    auto activation = get_activation_function(this->activate);
    for (auto i = 0; i < this->layers_num; i++) {
        z = this->weights[i] * z + this->biases[i];
        z.transform(activation);
    }
    z = this->weights[this->layers_num] * z;
    z.transform(activation);
    z = z - test_point.answer;

    double loss = 0.0;
    z.for_each([&loss](double x) {
        loss += x * x;
    });
    return loss / 2.0;
}

double ANN::train(const std::vector<TestPoint> &tps) {
    double total_loss = 0.0;

    for (TestPoint test_point : tps) {
        // z: before activation
        // a: after activation
        std::vector<arma::vec> z;
        std::vector<arma::vec> a;

        // forward propagation
        forward_propagation(z, a, test_point);

        // calculate loss
        double current_loss = 0.0;
        arma::vec error = a[a.size() - 1] - test_point.answer;
        error.for_each([&current_loss](double x) {
            current_loss += x * x;
            });
        current_loss /= 2.0;

        // back propagation
        back_propagation(z, a, test_point);
        total_loss += current_loss;
    }

    return total_loss / tps.size();
}

void ANN::save(const std::string& path) {
    nlohmann::json data = {
        { "layers_num", this->layers_num },
        { "layers_sizes", this->layers_sizes },
        { "weights", this->weights },
        { "biases", this->biases }
    };
    std::ofstream ofs(path);
    ofs << data.dump();
    ofs.close();
}

void ANN::load(const std::string& path) { // Not Implemented
    std::ifstream ifs(path);
    nlohmann::json j = nlohmann::json::parse(ifs);

    // this->layers_num = j["layers_num"];
    // this->layers_sizes = j["layers_sizes"];
    // this->weights = j["weights"];
    // this->biases = j["biases"];

    ifs.close();
}

void ANN::forward_propagation(std::vector<arma::vec>& z, std::vector<arma::vec>& a, const TestPoint& test_point) {
    auto activation = get_activation_function(this->activate);
    arma::vec current = test_point.input;
    a.push_back(current);
    for (auto i = 0; i < this->layers_num; i++) {
        current = this->weights[i] * current + this->biases[i];
        z.push_back(current);
        current.transform(activation);
        a.push_back(current);
    }
    current = this->weights[this->layers_num] * current;
    z.push_back(current);
    current.transform(activation);
    a.push_back(current);
}

void ANN::back_propagation(std::vector<arma::vec>& z, std::vector<arma::vec>& a, const TestPoint& test_point) {
    auto activation_derivative = get_activation_derivative_function(this->activate);
    std::vector<arma::vec> delta; // delta[0] meaning the delta of the last layer

    arma::vec activated_zl = z[z.size() - 1].transform(activation_derivative);
    delta.push_back((a[a.size() - 1] - test_point.answer) % activated_zl); // % element-wise multiplication

    for (auto i = this->layers_num - 1; i >= 0; i--) {
        activated_zl = z[i].transform(activation_derivative);
        delta.push_back((this->weights[i + 1].t() * delta[delta.size() - 1]) % activated_zl);
    }

    // update biases
    for (auto i = 0; i < this->layers_num; i++) {
        this->biases[i] -= this->alpha * delta[this->layers_num - i];
    }

    // update weights
    for (auto i = 0; i < this->layers_num; i++) {
        weights[i] -= this->alpha * delta[this->layers_num - i] * a[i].t();
    }
}
