#include "model/model.h"
#include "data/data.h"

Model::Model() {}

double Model::batch_test(Data &data, const int batch_size) {
    std::vector<TestPoint> batch_data;
    data.batch(batch_size, batch_data);

    double loss = 0.0;
    for (auto data: batch_data) {
        loss += this->single_test(data);
    }

    return loss;
}
