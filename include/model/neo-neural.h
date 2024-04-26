#pragma once

#include "model/model.h"

class NeoNeural: public Model {
public: 
    NeoNeural();
    ~NeoNeural();
    
    double single_test(const TestPoint &test_point); // return the loss
    double train(Data &data, int batch_size);

    void save(const std::string &path);
    void load(const std::string &path);
};
