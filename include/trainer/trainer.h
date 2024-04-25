#pragma once

#include <model/model.h>

void train(Model &model, Data &train_data, int batch_size, Data &test_data, int test_size, int epoch);
