#include "Regression.hpp"
#include "../Dataset/Dataset.hpp"

Regression::Regression(Dataset* X, Dataset* y) {
    m_X = X;
    m_y = y;
}

Dataset* Regression::get_y() const {
    return m_y;
}

Dataset* Regression::get_X() const {
    return m_X;
}
