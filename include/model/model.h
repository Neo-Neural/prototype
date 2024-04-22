#pragma once

#include "data/data.h"

class Model {
public:
    Model();
    
    double batch_test(Data &data, const int batch_size);
    virtual double single_test(const TestPoint &test_point) = 0; // return the loss
    virtual void train(const Data &data) {}

    virtual void save(const std::string &path) {}
    virtual void load(const std::string &path) {}

};
