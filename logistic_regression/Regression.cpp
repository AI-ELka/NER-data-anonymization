#include "Regression.hpp"
#include "Dataset.hpp"

Regression::Regression(Dataset* X, Dataset* y) {
    X = X;
    y = y;
}

Dataset* Regression::get_y() const {
    return y;
}

Dataset* Regression::get_X() const {
    return X;
}
