#pragma once

#include "data/data.h"

typedef int (*ClassificationFunction)(double x, double y);

class ClassifierData: public Data {
/**
 *  2d-classification task
 *  input dimension: 2
 *  output dimension: 2
 *  data type: double
 */
public:
    ClassifierData(ClassificationFunction classifier, int output_dim);
    void batch(int batch_size, std::vector<TestPoint>& data);

    // common classifiers
    static int circle_classifier(double x, double y);
    static int xor_classifier(double x, double y);
    static int square_classifier(double x, double y);
    static int dimension_classifier(double x, double y);

private:
    const int seed = 114514;
    const int input_dim = 2;
    
    int output_dim;
    ClassificationFunction classifier;
};
