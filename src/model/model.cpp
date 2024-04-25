#include "model/model.h"
#include "data/data.h"

// Model::Model() {}

double Model::batch_test(const std::vector<TestPoint> &test_point) {
    double total_loss = 0.0;
    for (auto &tp: test_point) {
        total_loss += this->single_test(tp);
    }
    return total_loss / test_point.size();
}
