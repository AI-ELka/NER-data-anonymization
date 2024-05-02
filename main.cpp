#include <iostream>
#include <Eigen/Dense>
#include "Dataset/Dataset.hpp"
#include <vector>
#include "logistic_regression/LogisticRegression.hpp"
#include "logistic_regression/MulticlassClassifier.hpp"

int main() {
    // Load the dataset
    Dataset X_train("data/representation.eng.testa.csv");
    Dataset y_train("data/true_labels.eng.testa.csv", false, true);

    
    // Parameters for logistic regression
    double lr = 0.01;
    long m_epochs = 100;

    // Encode the labels
    std::vector<std::string> labels = {"I-PER", "O", "I-MISC", "I-LOC" "I-ORG"};

    // Create an instance of MulticlassClassifier and fit the model
    MulticlassClassifier classifier(&X_train, &y_train, lr, m_epochs, labels);

    // Accuracy on the test set a
    Dataset X_test_a("data/representation.eng.testa.csv");
    Dataset y_test_a("data/true_labels.eng.testa.csv", false, true);

    std::cout << "Accuracy on test set a: " << classifier.accuracy(X_test_a, y_test_a) << std::endl;

    return 0;

}

// int main() {
//     // Load the dataset
//     Dataset X_train("data/representation.eng.testa.csv");
//     Dataset y_train("data/true_labels.eng.testa.csv", true);
    
//     // Parameters for logistic regression
//     double lr = 0.001;
//     long m_epochs = 100;

//     // Create an instance of LogisticRegression and fit the model
//     LogisticRegression logReg(&X_train, &y_train, lr, m_epochs);

//     // Accuracy on the training set
//     std::cout << "Accuracy on training set: " << logReg.accuracy(X_train, y_train) << std::endl;
    
    
//     // Accuracy on the test set a
//     Dataset X_test_a("data/representation.eng.testa.csv");
//     Dataset y_test_a("data/true_labels.eng.testa.csv", true);

//     std::cout << "Accuracy on test set a: " << logReg.accuracy(X_test_a, y_test_a) << std::endl;

//     // Accuracy on the test set b
//     Dataset X_test_b("data/representation.eng.testb.csv");
//     Dataset y_test_b("data/true_labels.eng.testb.csv", true);

//     std::cout << "Accuracy on test set b: " << logReg.accuracy(X_test_b, y_test_b) << std::endl;

//     return 0;
// }
