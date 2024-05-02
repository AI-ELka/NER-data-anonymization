#include <iostream>
#include "Dataset/Dataset.hpp"
#include "logistic_regression/LogisticRegression.hpp"
using namespace std;

int main() {
    Dataset X("data/representation.eng.train.csv");
    cout << "Number of samples: " << X.get_nbr_samples() << endl;
    cout << "Number of dimensions: " << X.get_dim() << endl;

    Dataset y("data/true_labels.eng.train.csv", true);
    cout << "Number of samples: " << y.get_nbr_samples() << endl;
    cout << "Number of dimensions: " << y.get_dim() << endl;

    LogisticRegression logistic_regression(&X, &y, 0.01, 1000);
    logistic_regression.show_coefficients();
    return 0;
}