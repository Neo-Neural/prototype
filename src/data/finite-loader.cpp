#include "data/finite-loader.h"
#include "data/data.h"
#include <vector>

FiniteLoaderData::FiniteLoaderData(std::vector<TestPoint> &tps) {
    this->tps.assign(tps.begin(), tps.end());
}

FiniteLoaderData::FiniteLoaderData(Loader loader) {
    loader(tps);
}

FiniteLoaderData::~FiniteLoaderData() {}

bool FiniteLoaderData::batch(int batch_size, std::vector<TestPoint> &data) {
    int data_num = this->tps.size();
    if (batch_size > data_num) {
        return false;
    }
    if (this->dataptr + batch_size > data_num) {
        // simply ignores it
        this->dataptr = 0;
    }
    data.assign(this->tps.begin() + this->dataptr, this->tps.begin() + this->dataptr + batch_size);
    this->dataptr += batch_size;
    return true;
}
