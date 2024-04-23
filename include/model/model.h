#pragma once

#include "data/data.h"

class Model {
public:
    Model();
    
    double batch_test(Data &data, const int batch_size);
    virtual double single_test(const TestPoint &test_point) = 0; // return the loss
    virtual double train(Data& data, int epoch, int batch_size) = 0; // return train loss

    virtual void save(const std::string &path) {}
    virtual void load(const std::string &path) {}

};
