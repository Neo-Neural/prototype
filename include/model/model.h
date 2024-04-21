#pragma once

#include "data/data.h"

class Model {
public:
    Model();
    
    virtual double test(const Data &data); // return the loss
    virtual void train(const Data &data);

    virtual void save(const std::string &path);
    virtual void load(const std::string &path);

};
