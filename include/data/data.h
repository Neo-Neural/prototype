#pragma once

#include <armadillo>
#include <vector>

struct TestPoint {
    arma::vec input;
    arma::vec answer;
};

class Data {
public:
    virtual ~Data() = default;

    virtual bool batch(int batch_size, std::vector<TestPoint>& data) = 0;
    // return whether the data is successfully loaded
};
