#include <model/ann.h>
#include <data/classifier.h>
#include <trainer/trainer.h>

#include "data/finite-loader.h"

int main() {
    FiniteLoaderData train_data([](std::vector<TestPoint> &tps) {
        ClassifierData generator(ClassifierData::circle_classifier, 2);
        generator.batch(1000, tps);
    });

    ClassifierData test_data(ClassifierData::circle_classifier, 2);

    std::vector<int> hidden = {5, 3, 6};
    ANN model(2, hidden, 2);

    train(model, train_data, 1000, test_data, 10, 5000);
    return 0;
}
