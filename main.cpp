#include <iostream>
#include <Eigen/Dense>
#include "Dataset/Dataset.hpp"
#include "logistic_regression/LogisticRegression.hpp"

int main() {
    // Load the dataset
    Dataset X("/home/mach/Desktop/info432/Data_anonymization/NER_Project/data/representation.eng.testa.csv");  // Assuming the CSV is properly formatted for features
    Dataset y("/home/mach/Desktop/info432/Data_anonymization/NER_Project/data/true_labels.eng.testa.csv", true); // Assuming second parameter 'true' signifies loading labels
    // Parameters for logistic regression
    double learningRate = 0.01;
    long epochs = 1000;

    // Create an instance of LogisticRegression
    LogisticRegression logReg(&X, &y, learningRate, epochs);

    std::cout << "Logistic Regression model created." << std::endl;


    
    // Fit the model
    // logReg.set_coefficients();

    // Display coefficients
    // logReg.show_coefficients();

    // Example of using the model to predict
    // Here, we might take a random instance from `X` and estimate the output
    const std::vector<double>& instance = X.get_instance(0);  // get the first instance
    Eigen::VectorXd vec(instance.size());
    for (size_t i = 0; i < instance.size(); ++i) {
        vec(i) = instance[i];
    }
    double prediction = logReg.estimate(vec);
    std::cout << "Predicted value for the first instance: " << prediction << std::endl;
    // logistic_regression.show_coefficients();
    return 0;
}
