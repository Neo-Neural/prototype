#pragma once

#include "model/model.h"

class ANN: public Model {

public:
    ANN(const int input_dim, const std::vector<int> &layers, const int output_dim);
    
    double single_test(const TestPoint &test_point);
    double train(Data& data, int epoch, int batch_size = 1);

    void save(const std::string &path);
    void load(const std::string &path);

private:
    int layers_num;
    std::vector<int> layers_sizes;
    std::vector<arma::mat> weights; // weights[i] - weights between layers[i] and layers[i+1]
    std::vector<arma::vec> biases; // biases[i] - biases for layer[i+1]

    const double alpha = 0.001; // learning rate

    //z before activation;a after
    void forward_propagation(std::vector<arma::vec>& z, std::vector<arma::vec>& a, const TestPoint& test_point);
    void back_propagation(std::vector<arma::vec>& z, std::vector<arma::vec>& a, const TestPoint& test_point);

};