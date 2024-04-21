#pragma once

#include "model/model.h"

class NeoNeural: public Model {
public: 
    NeoNeural();
    double test(const Data &data); // return the loss
    void train(const Data &data);

    void save(const std::string &path);
    void load(const std::string &path);
};
