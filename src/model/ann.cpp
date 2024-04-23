#include "model/ann.h"

#include "data/data.h"
#include "util/activation.h"

#include <random>
#include <vector>

ANN::ANN(const int input_dim, const std::vector<int>& layers, const int output_dim) {
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

double ANN::single_test(const TestPoint& test_point) {
    arma::vec z(test_point.input);
    for (auto i = 0; i < this->layers_num; i++) {
        z = this->weights[i] * z + this->biases[i];
        z.transform(Activation::tanh);
    }
    z = this->weights[this->layers_num] * z;
    z.transform(Activation::tanh);
    z = z - test_point.answer;

    double loss = 0.0;
    z.for_each([&loss](double x) {
        loss += x * x;
        });
    return loss / 2.0;
}

void ANN::forward_propagation(std::vector<arma::vec>& z, std::vector<arma::vec>& a, const TestPoint& test_point) {
    arma::vec current = test_point.input;
    a.push_back(current);
    for (auto i = 0; i < this->layers_num; i++) {
        current = this->weights[i] * current + this->biases[i];
        z.push_back(current);
        current.transform(Activation::tanh);
        a.push_back(current);
    }
    current = this->weights[this->layers_num] * current;
    z.push_back(current);
    current.transform(Activation::tanh);
    a.push_back(current);
}

void ANN::back_propagation(std::vector<arma::vec>& z, std::vector<arma::vec>& a, const TestPoint& test_point) {
    std::vector<arma::vec> delta; // delta[0] meaning the delta of the last layer

    arma::vec activated_zl = z[z.size() - 1].transform(ActivationDifferential::tanh);
    delta.push_back((a[a.size() - 1] - test_point.answer) % activated_zl);//% element-wise multiplication

    for (auto i = this->layers_num - 1; i >= 0; i--) {
        activated_zl = z[i].transform(ActivationDifferential::tanh);
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

double ANN::train(Data& data, int epoch, int batch_size) {
    double minloss = 100.0; // a big enough value

    std::vector<TestPoint> tps;
    for (int i = 1; i <= epoch; i++) {
        double loss = 0.0;
        int batch_i = 0;

        while (data.batch(batch_size, tps)) {
            double batch_loss = 0.0;
            batch_i ++;

            for (TestPoint test_point : tps) {
                // z: before activation
                // a: after activation
                std::vector<arma::vec> z;
                std::vector<arma::vec> a;

                // forward propagation
                forward_propagation(z, a, test_point);

                //calculate loss
                double current_loss = 0.0;
                arma::vec error = a[a.size() - 1] - test_point.answer;
                error.for_each([&current_loss](double x) {
                    current_loss += x * x;
                    });
                current_loss /= 2.0;

                // back propagation
                back_propagation(z, a, test_point);

                batch_loss += current_loss;
            }
            batch_loss /= batch_size;

            // later update params from it
            loss += batch_loss;
        }
        loss /= batch_i;
        printf("Epoch: %4d    Loss: %f\n", i, loss);
        minloss = std::min(minloss, loss);
    }

    return minloss;
}

void ANN::save(const std::string& path) {

}

void ANN::load(const std::string& path) {

}
