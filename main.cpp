#include <model/ann.h>
#include <data/classifier.h>

int main() {
    ClassifierData data(ClassifierData::circle_classifier, 2);
    data.generateData(1000);
    /*
    std::vector<TestPoint>data1;
    
    int t = 30000;
    data.batch(t, data1);
    int an1 = 0;
    for (int i = 0; i < t; i++) {
        if (int(data1[i].answer[0]) == -1)an1++;
      //  printf("%f  %f   %f   %f\n", data1[i].input[0], data1[i].input[1], data1[i].answer[0], data1[i].answer[1]);
    }
    printf("%d\n", an1);*/

    std::vector<int> hidden = {5, 3, 6};
    ANN model(2, hidden, 2);
    model.train(data, 50, 10);
    data.reset_dataptr();
    printf("%f", model.batch_test(data, 500)/500.0);
    return 0;
}