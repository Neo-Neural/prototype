#include "data/data.h"
#include "fmt/core.h"
#include "fmt/color.h"
#include <trainer/trainer.h>
#include <vector>

void train(Model &model, Data &train_data, int batch_size, Data &test_data, int test_size, int epoch) {
    for (int i = 0; i < epoch; i++) {
        std::vector<TestPoint> tps;
        train_data.batch(batch_size, tps);
        double train_loss = model.train(tps);
        fmt::print(fmt::fg(fmt::color::green), "Epoch {}: train_loss = {:1.5}\n", i + 1, train_loss);

        test_data.batch(test_size, tps);
        double test_loss = model.batch_test(tps);
        fmt::print(fmt::fg(fmt::color::red), "Epoch {}:  test_loss = {:1.5}\n", i + 1, test_loss);
    }
}
