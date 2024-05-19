#include <model/ann.h>
#include <data/classifier.h>
#include <trainer/trainer.h>
#include <fmt/core.h>

#include "data/finite-loader.h"
#include "util/activation.h"

int main() {
    FiniteLoaderData train_data([](std::vector<TestPoint> &tps) {
        ClassifierData generator(ClassifierData::square_classifier, 2);
        generator.batch(1000, tps);
    });

    ClassifierData test_data(ClassifierData::square_classifier, 2);

    std::vector<int> hidden = {8, 12, 24, 12, 8};
    ANN model(2, hidden, 2, TANH);

    train(model, train_data, 500, test_data, 30, 5000);

    model.save("data.json");

    return 0;
}
