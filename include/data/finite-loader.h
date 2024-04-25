#pragma once

#include <data/data.h>
#include <vector>

typedef void (*Loader)(std::vector<TestPoint> &tps);

class FiniteLoaderData: public Data {
public:
    FiniteLoaderData(std::vector<TestPoint> &tps);
    FiniteLoaderData(Loader loader);
    ~FiniteLoaderData();

    bool batch(int batch_size, std::vector<TestPoint> &data);    

private:
    std::vector<TestPoint> tps;
    int dataptr = 0;
};
