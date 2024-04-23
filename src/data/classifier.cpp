#include "data/classifier.h"
#include "data/data.h"
#include <cstdlib>
#include <cstring>
#include <vector>
#include <armadillo>

ClassifierData::ClassifierData(
    ClassificationFunction classifier,
    int output_dim = 2
): classifier(classifier), output_dim(output_dim),
    datasize(0), dataptr(0) {
    srand(this->seed);
}

int ClassifierData::circle_classifier(double x, double y) {
    return x * x + y * y <= 1 ? 0 : 1;
}

int ClassifierData::xor_classifier(double x, double y) {
    return (x > 0) ^ (y > 0) ? 0 : 1;
}

int ClassifierData::square_classifier(double x, double y) {
    return fabs(x) <= 0.5 && fabs(y) <= 0.5 ? 0 : 1;
}

int ClassifierData::dimension_classifier(double x, double y) {
    switch ((x > 0 ? 1 : 0) + (y > 0 ? 2: 0)) {
        case 0: return 2;
        case 1: return 3;
        case 2: return 1;
        case 3: return 0;
    }
    return 1919810; // to make the compiler happy
}


int ClassifierData::batch(int batch_size, std::vector<TestPoint>& data) {
    data.clear();
    batch_size += dataptr;
    if (batch_size < datasize)
        for (; dataptr < batch_size; dataptr++)
            data.push_back(innerdata[dataptr]);
    else dataptr = 0;
    return dataptr;
}


void ClassifierData::generateData(int size) {
    datasize = size;
    dataptr = 0;
    for (auto i = 0; i < size; i++) {
        double x = rand() * 2.0 / RAND_MAX - 1;
        double y = rand() * 2.0 / RAND_MAX - 1;
        TestPoint tp;
        tp.input = arma::vec({ x, y });
        int result = this->classifier(x, y);
        tp.answer = arma::vec(this->output_dim);
        tp.answer.fill(-1.0);
        tp.answer[result] = 1.0;
        innerdata.push_back(tp);
    }
}