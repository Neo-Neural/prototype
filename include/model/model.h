#pragma once

#include "data/data.h"

class Model {
public:
    virtual ~Model() = default;

    double batch_test(const std::vector<TestPoint> &test_point);

    virtual double single_test(const TestPoint test_point) = 0; // return the loss
    virtual double train(const std::vector<TestPoint> &tps) = 0; // return train loss

    virtual void save(const std::string &path) = 0;
    virtual void load(const std::string &path) = 0;

};
