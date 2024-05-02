#include <iostream>
#include <Eigen/Dense>
#include "Dataset/Dataset.hpp"
#include "logistic_regression/LogisticRegression.hpp"

int main() {
    // Load the dataset
    Dataset X_train("data/representation.eng.testa.csv");
    Dataset y_train("data/true_labels.eng.testa.csv", true);
    
    // Parameters for logistic regression
    double lr = 0.01;
    long m_epochs = 1000;

    // Create an instance of LogisticRegression and fit the model
    LogisticRegression logReg(&X_train, &y_train, lr, m_epochs);
    
    
    // Example of using the model to predict
    Dataset X_test_a("data/representation.eng.testa.csv");
    Dataset y_test_a("data/true_labels.eng.testa.csv", true);
    long S = 0;
    for (int i = 0; i < X_test_a.get_nbr_samples(); i++) {
        const std::vector<double> instance = X_test_a.get_instance(i);
        Eigen::VectorXd vec(instance.size());
        
        for (size_t i = 0; i < instance.size(); ++i) {
            vec(i) = instance[i];
        }

        double prediction = logReg.estimate(vec);
        if (prediction >= 0.5 && y_test_a.get_instance(i)[0] == 1) {
            S++;
        } else if (prediction < 0.5 && y_test_a.get_instance(i)[0] == 0) {
            S++;
        }
    }
    std::cout << "Accuracy: " << S / (double)X_test_a.get_nbr_samples() << std::endl;

    return 0;
}
