#include <iostream>
#include "dataset/Dataset.hpp"
using namespace std;

int main() {
    Dataset dataset("data/representation.eng.train.csv");
    cout << "Number of samples: " << dataset.get_nbr_samples() << endl;
    cout << "Number of dimensions: " << dataset.get_dim() << endl;
    return 0;
}