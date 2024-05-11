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
        std::cout << "\nFitting the model..." << std::endl;
        LogisticRegression logReg(&X_train, &y_train, lr, m_epochs);
        std::cout << "Model fitted.\n" << std::endl;

        // Accuracy on the training set
        std::cout << "Accuracy on training set: " << logReg.accuracy(X_train, y_train) << std::endl;
        std::cout << "Precision on training set: " << logReg.precision(X_train, y_train) << std::endl;
        std::cout << "Recall on training set: " << logReg.recall(X_train, y_train) << std::endl;
        std::cout << "F1 score on training set: " << logReg.f1_score(X_train, y_train) << std::endl;
        
        // Accuracy on the test set a
        Dataset X_test_a("data/representation.eng.testa.csv");
        Dataset y_test_a("data/true_labels.eng.testa.csv", true);
        std::cout << "\nEvaluating the model on the test set a..." << std::endl;

        std::cout << "Accuracy on test set a: " << logReg.accuracy(X_test_a, y_test_a) << std::endl;
        std::cout << "Precision on test set a: " << logReg.precision(X_test_a, y_test_a) << std::endl;
        std::cout << "Recall on test set a: " << logReg.recall(X_test_a, y_test_a) << std::endl;
        std::cout << "F1 score on test set a: " << logReg.f1_score(X_test_a, y_test_a) << std::endl;

        // Accuracy on the test set b
        Dataset X_test_b("data/representation.eng.testb.csv");
        Dataset y_test_b("data/true_labels.eng.testb.csv", true);
        std::cout << "\nEvaluating the model on the test set b..." << std::endl;

        std::cout << "Accuracy on test set b: " << logReg.accuracy(X_test_b, y_test_b) << std::endl;
        std::cout << "Precision on test set b: " << logReg.precision(X_test_b, y_test_b) << std::endl;
        std::cout << "Recall on test set b: " << logReg.recall(X_test_b, y_test_b) << std::endl;
        std::cout << "F1 score on test set b: " << logReg.f1_score(X_test_b, y_test_b) << std::endl;
    } 
    else if (process == "multiclass") 
    {
        // Encode the labels
        vector<string> labels = {"O", "PER", "LOC", "MISC"};;

        // Load the dataset
        Dataset X_train("data/representation.eng.train.csv");
        Dataset y_train("data/true_labels.eng.train.csv", false, true);
        
        // Parameters for logistic regression for multiclass classification
        double lr = 0.001;
        long m_epochs = 500;

        // Create an instance of MulticlassClassifier and fit the model
        cout << "\nFitting the model..." << endl;
        MulticlassClassifier classifier(&X_train, &y_train, lr, m_epochs, "one_vs_one");
        cout << "Model fitted-----------------.ONEVONE-----------------------\n" << endl;

        // metrics on the training set
        classifier.show_confusion_matrix(X_train, y_train);
        cout << "Accuracy on training set: " << classifier.accuracy(X_train, y_train) << endl;
        cout << "Precision on training set: " << classifier.precision(X_train, y_train) << endl;
        cout << "Recall on training set: " << classifier.recall(X_train, y_train) << endl;
        cout << "F1 score on training set: " << classifier.f1_score(X_train, y_train) << endl;

        // metrics on the test set a
        cout << "\nEvaluating the model on the test set------------- BTC--------- a..." << endl;
        Dataset X_test_a("data/representation.e.conll.csv");
        Dataset y_test_a("data/true_labels.e.conll.csv", false, true);

        classifier.show_confusion_matrix(X_test_a, y_test_a);
        cout << "Accuracy on test set a: " << classifier.accuracy(X_test_a, y_test_a) << endl;
        cout << "Precision on test set a: " << classifier.precision(X_test_a, y_test_a) << endl;
        cout << "Recall on test set a: " << classifier.recall(X_test_a, y_test_a) << endl;
        cout << "F1 score on test set a: " << classifier.f1_score(X_test_a, y_test_a) << endl;

        // metrics on the test set a
        cout << "\nEvaluating the model on the test set -----------WIKIGOLD-------- a..." << endl;
        Dataset X_test_b("data/representation.wikigold.conll.txt.csv");
        Dataset y_test_b("data/true_labels.wikigold.conll.txt.csv", false, true);

        classifier.show_confusion_matrix(X_test_b, y_test_b);
        cout << "Accuracy on test set a: " << classifier.accuracy(X_test_b, y_test_b) << endl;
        cout << "Precision on test set a: " << classifier.precision(X_test_b, y_test_b) << endl;
        cout << "Recall on test set a: " << classifier.recall(X_test_b, y_test_b) << endl;
        cout << "F1 score on test set a: " << classifier.f1_score(X_test_b, y_test_b) << endl;

        cout << "-----------------~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$$$$$$$$$$$$$$$$$$$$$$$$"<<endl;

        // Load the dataset
        Dataset X_train2("data/representation.eng.train.csv");
        Dataset y_train2("data/true_labels.eng.train.csv", false, true);
        

        // Create an instance of MulticlassClassifier and fit the model
        cout << "\nFitting the model..." << endl;
        MulticlassClassifier classifier2(&X_train2, &y_train2, lr, m_epochs, "one_vs_all");
        cout << "Model fitted-----------------.ONEVALLLLLLLL-----------------------\n" << endl;

        // metrics on the training set
        classifier2.show_confusion_matrix(X_train2, y_train2);
        cout << "Accuracy on training set: " << classifier2.accuracy(X_train2, y_train2) << endl;
        cout << "Precision on training set: " << classifier2.precision(X_train2, y_train2) << endl;
        cout << "Recall on training set: " << classifier2.recall(X_train2, y_train2) << endl;
        cout << "F1 score on training set: " << classifier2.f1_score(X_train2, y_train2) << endl;

        // metrics on the test set a
        cout << "\nEvaluating the model on the test set------------- BTC--------- a..." << endl;
        Dataset X_test_a2("data/representation.e.conll.csv");
        Dataset y_test_a2("data/true_labels.e.conll.csv", false, true);

        classifier2.show_confusion_matrix(X_test_a2, y_test_a2);
        cout << "Accuracy on test set a: " << classifier2.accuracy(X_test_a2, y_test_a2) << endl;
        cout << "Precision on test set a: " << classifier2.precision(X_test_a2, y_test_a2) << endl;
        cout << "Recall on test set a: " << classifier2.recall(X_test_a2, y_test_a2) << endl;
        cout << "F1 score on test set a: " << classifier2.f1_score(X_test_a2, y_test_a2) << endl;

        // metrics on the test set a
        cout << "\nEvaluating the model on the test set -----------WIKIGOLD-------- a..." << endl;
        Dataset X_test_b2("data/representation.wikigold.conll.txt.csv");
        Dataset y_test_b2("data/true_labels.wikigold.conll.txt.csv", false, true);

        classifier2.show_confusion_matrix(X_test_b2, y_test_b2);
        cout << "Accuracy on test set a: " << classifier2.accuracy(X_test_b2, y_test_b2) << endl;
        cout << "Precision on test set a: " << classifier2.precision(X_test_b2, y_test_b2) << endl;
        cout << "Recall on test set a: " << classifier2.recall(X_test_b2, y_test_b2) << endl;
        cout << "F1 score on test set a: " << classifier2.f1_score(X_test_b2, y_test_b2) << endl;
    } else {
        cout << "Invalid input !" << endl;
    }
    return 0;
}
