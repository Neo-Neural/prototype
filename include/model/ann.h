#pragma once

#include "model/model.h"

class ANN: public Model {

public:
    ANN(const std::string &path);
    ANN(const int input_dim, const std::vector<int> &layers, const int output_dim);
    ~ANN();

    double single_test(const TestPoint &test_point);
    double train(const std::vector<TestPoint> &tps);

    void save(const std::string &path);
    void load(const std::string &path);

private:
    int layers_num;
    std::vector<int> layers_sizes;
    std::vector<arma::mat> weights; // weights[i] - weights between layers[i] and layers[i+1]
    std::vector<arma::vec> biases; // biases[i] - biases for layer[i+1]

    const double alpha = 0.001; // learning rate

    // z: before activation
    // a: after activation
    void forward_propagation(std::vector<arma::vec>& z, std::vector<arma::vec>& a, const TestPoint& test_point);
    void back_propagation(std::vector<arma::vec>& z, std::vector<arma::vec>& a, const TestPoint& test_point);

};