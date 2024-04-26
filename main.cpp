#include <model/ann.h>
#include <data/classifier.h>
#include <trainer/trainer.h>

#include "data/finite-loader.h"

int main() {
    FiniteLoaderData train_data([](std::vector<TestPoint> &tps) {
        ClassifierData generator(ClassifierData::dimension_classifier, 4);
        generator.batch(1000, tps);
    });

    ClassifierData test_data(ClassifierData::dimension_classifier, 4);

    std::vector<int> hidden = {8, 12, 12, 12, 8};
    ANN model(2, hidden, 4);

    train(model, train_data, 1000, test_data, 30, 500);

    model.save("data.json");

    return 0;
}
