#include <model/ann.h>
#include <data/classifier.h>

int main() {
    ClassifierData data(ClassifierData::circle_classifier, 2);
    data.generateData(1000);

    std::vector<int> hidden = {5, 3, 6};
    ANN model(2, hidden, 2);
    model.train(data, 50, 10);
    data.reset_dataptr();
    printf("%f", model.batch_test(data, 500)/500.0);
    return 0;
}