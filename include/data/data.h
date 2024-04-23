#pragma once

#include <armadillo>
#include <vector>

struct TestPoint {
    arma::vec input;
    arma::vec answer;
};

class Data {
public:
    Data();
    virtual int batch(int batch_size, std::vector<TestPoint>& data) = 0;//return the dataptr for batch
};
