#include <iostream>
#include <Eigen/Dense>
#include "Dataset/Dataset.hpp"
#include <vector>
#include "logistic_regression/LogisticRegression.hpp"
#include "logistic_regression/MulticlassClassifier.hpp"
#include <string>

int main() {
    string process;
    cout << "Enter 'binary' for binary classification or 'multiclass' for multiclass classification: ";
    cin >> process;
    
    if (process == "binary")
    {
        // Load the dataset
        Dataset X_train("data/representation.eng.train.csv");
        Dataset y_train("data/true_labels.eng.train.csv", true, false);
        
        // Parameters for logistic regression
        double lr = 0.001;
        long m_epochs = 100;

        // Create an instance of LogisticRegression and fit the model.
        LogisticRegression logReg(&X_train, &y_train, lr, m_epochs);

        // Accuracy on the training set
        std::cout << "Accuracy on training set: " << logReg.accuracy(X_train, y_train) << std::endl;
        std::cout << "Precision on training set: " << logReg.precision(X_train, y_train) << std::endl;
        std::cout << "Recall on training set: " << logReg.recall(X_train, y_train) << std::endl;
        std::cout << "F1 score on training set: " << logReg.f1_score(X_train, y_train) << std::endl;
        
        // Accuracy on the test set a
        Dataset X_test_a("data/representation.eng.testa.csv");
        Dataset y_test_a("data/true_labels.eng.testa.csv", true);

        std::cout << "Accuracy on test set a: " << logReg.accuracy(X_test_a, y_test_a) << std::endl;
        std::cout << "Precision on test set a: " << logReg.precision(X_test_a, y_test_a) << std::endl;
        std::cout << "Recall on test set a: " << logReg.recall(X_test_a, y_test_a) << std::endl;
        std::cout << "F1 score on test set a: " << logReg.f1_score(X_test_a, y_test_a) << std::endl;

        // Accuracy on the test set b
        Dataset X_test_b("data/representation.eng.testb.csv");
        Dataset y_test_b("data/true_labels.eng.testb.csv", true);

        std::cout << "Accuracy on test set b: " << logReg.accuracy(X_test_b, y_test_b) << std::endl;
        std::cout << "Precision on test set b: " << logReg.precision(X_test_b, y_test_b) << std::endl;
        std::cout << "Recall on test set b: " << logReg.recall(X_test_b, y_test_b) << std::endl;
        std::cout << "F1 score on test set b: " << logReg.f1_score(X_test_b, y_test_b) << std::endl;
    } 
    else if (process == "multiclass") 
    {
        // Load the dataset
        Dataset X_train("data/representation.eng.testa.csv");
        Dataset y_train("data/true_labels.eng.testa.csv", false, true);
        
        // Parameters for logistic regression for multiclass classification
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
    } else {
        cout << "Invalid input !" << endl;
    }
    return 0;
}