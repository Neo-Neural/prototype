#include <model/ann.h>
#include <data/classifier.h>

int main() {
    ClassifierData data(ClassifierData::circle_classifier, 2);

    std::vector<int> hidden = {5, 5, 5};
    ANN model(2, hidden, 2);

    printf("%f", model.batch_test(data, 20));
    return 0;
}