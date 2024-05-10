<<<<<<< HEAD
#include <iostream>
#include <cassert>
=======
#include<iostream>
#include<cassert>
>>>>>>> origin/main
#include "../Dataset/Dataset.hpp"
#include "LogisticRegression.hpp"
#include "../Dataset/Dataset.hpp"
#include "Regression.hpp"

<<<<<<< HEAD
using namespace std;

LogisticRegression::LogisticRegression(Dataset *X, Dataset *y, double lr, long m_epochs) : Regression(X, y)
{
    m_beta = nullptr;
    learning_rate = lr;
    epochs = m_epochs;
    set_coefficients();
}

LogisticRegression::~LogisticRegression()
{
    if (m_beta != nullptr)
    {
=======

LogisticRegression::LogisticRegression(Dataset* X, Dataset* y, double learning_rate, long epochs) : Regression(X, y) {
	m_beta = NULL;
	this->learning_rate = learning_rate;
	this->epochs = epochs;
    std::cout << "Setting coeffs" << std::endl;
	set_coefficients();
}

LogisticRegression::~LogisticRegression() {
    if (m_beta != nullptr) {
>>>>>>> origin/main
        delete m_beta;
        m_beta = nullptr;
    }
}

<<<<<<< HEAD
Eigen::MatrixXd LogisticRegression::construct_matrix()
{
    Eigen::MatrixXd Xones(get_X()->get_nbr_samples(), get_X()->get_dim() + 1);
    Xones.col(0).setOnes();
    for (int i = 0; i < get_X()->get_nbr_samples(); i++)
    {
        for (int j = 0; j < get_X()->get_dim(); j++)
        {
=======
Eigen::MatrixXd LogisticRegression::construct_matrix() {
    Eigen::MatrixXd Xones(get_X()->get_nbr_samples(), get_X()->get_dim() + 1);
    Xones.col(0).setOnes();
    for (int i = 0; i < get_X()->get_nbr_samples(); i++) {
        for (int j = 0; j < get_X()->get_dim(); j++) {
>>>>>>> origin/main
            Xones(i, j + 1) = get_X()->get_instance(i)[j];
        }
    }
    return Xones;
}

<<<<<<< HEAD
Eigen::VectorXd LogisticRegression::construct_y()
{
    Eigen::VectorXd y(get_y()->get_nbr_samples());
    for (int i = 0; i < get_y()->get_nbr_samples(); i++)
    {
        y(i) = get_y()->get_instance(i)[0];
    }
    return y;
}

void LogisticRegression::set_coefficients()
{
    Eigen::MatrixXd X = construct_matrix();
    Eigen::VectorXd y = construct_y();

    m_beta = new Eigen::VectorXd(X.cols());
    m_beta->setZero();
    for (int i = 0; i < epochs; i++)
    {
        *m_beta += learning_rate * gradient(X, y, *m_beta);
    }
}

const Eigen::VectorXd *LogisticRegression::get_coefficients() const
{
    if (!m_beta)
    {
        cout << "Coefficients have not been allocated." << endl;
=======
Eigen::VectorXd LogisticRegression::construct_y() {
    Eigen::VectorXd y(get_y()->get_nbr_samples());
    for (int i = 0; i < get_y()->get_nbr_samples(); i++) {
        y(i) = get_y()->get_instance(i)[0];
    }

    return y;
}

void LogisticRegression::set_coefficients() {
    Eigen::MatrixXd X = construct_matrix();
    Eigen::VectorXd y = construct_y();
    std::cout << "X rows then columns " << X.rows()<< "  "<< X.cols() << std::endl;
    std::cout << "y rows then columns " << y.rows()<< "  "<< y.cols() << std::endl;
    m_beta = new Eigen::VectorXd(X.cols());
    m_beta->setZero();
    for (int i = 0; i < epochs; i++) {
        *m_beta -= learning_rate * gradient(X, y);
    }
}

const Eigen::VectorXd* LogisticRegression::get_coefficients() const {
    if (!m_beta) {
        std::cout << "Coefficients have not been allocated." << std::endl;
>>>>>>> origin/main
        return nullptr;
    }
    return m_beta;
}

<<<<<<< HEAD
double LogisticRegression::estimate(const Eigen::VectorXd &x) const
{
=======
void LogisticRegression::show_coefficients() const {
    if (!m_beta) {
        std::cout << "Coefficients have not been allocated." << std::endl;
        return;
    }

    if (m_beta->size() != m_X->get_dim() + 1) {
        std::cout << "Warning, unexpected size of coefficients vector: " << m_beta->size() << std::endl;
    }

    std::cout << "beta = (";
    for (int i = 0; i < m_beta->size(); i++) {
        std::cout << " " << (*m_beta)[i];
    }
    std::cout << " )" << std::endl;
}

void LogisticRegression::print_raw_coefficients() const {
    std::cout << "{ ";
    for (int i = 0; i < m_beta->size() - 1; i++) {
        std::cout << (*m_beta)[i] << ", ";
    }
    std::cout << (*m_beta)[m_beta->size() - 1];
    std::cout << " }" << std::endl;
}

double LogisticRegression::estimate(const Eigen::VectorXd & x) const {
>>>>>>> origin/main
    double S = sigmoid((*m_beta)(0) + x.transpose() * m_beta->tail(m_beta->size() - 1));
    return S;
}

<<<<<<< HEAD
double LogisticRegression::sigmoid(double x) const
{
    return 1 / (1 + exp(-x));
}

Eigen::VectorXd LogisticRegression::gradient(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, Eigen::VectorXd &beta) const
{
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(X.cols());
    for (int i = 0; i < X.rows(); i++)
    {
        grad += (y(i) - sigmoid(X.row(i).dot(beta))) * X.row(i);
    }
    return grad.transpose();
}

void LogisticRegression::confusion_matrix(const Dataset &X_test, const Dataset &y_test, Eigen::MatrixXd &con_matrix) const
{
    con_matrix = Eigen::MatrixXd::Zero(2, 2);
    for (int i = 0; i < X_test.get_nbr_samples(); i++)
    {
        const std::vector<double> instance = X_test.get_instance(i);
        Eigen::VectorXd vec(instance.size());
        for (size_t i = 0; i < instance.size(); ++i)
        {
            vec(i) = instance[i];
        }
        double prediction = estimate(vec);

        if (prediction >= 0.5 && y_test.get_instance(i)[0] == 1)
        {
            con_matrix(0, 0)++;
        }
        else if (prediction < 0.5 && y_test.get_instance(i)[0] == 0)
        {
            con_matrix(1, 1)++;
        }
        else if (prediction >= 0.5 && y_test.get_instance(i)[0] == 0)
        {
            con_matrix(1, 0)++;
        }
        else if (prediction < 0.5 && y_test.get_instance(i)[0] == 1)
        {
            con_matrix(0, 1)++;
        }
    }
}

double LogisticRegression::accuracy(const Dataset &X_test, const Dataset &y_test) const
{
    Eigen::MatrixXd con_matrix;
    confusion_matrix(X_test, y_test, con_matrix);
    return (con_matrix(0, 0) + con_matrix(1, 1)) / (con_matrix.sum());
}

double LogisticRegression::precision(const Dataset &X_test, const Dataset &y_test) const
{
    Eigen::MatrixXd con_matrix;
    confusion_matrix(X_test, y_test, con_matrix);
    return con_matrix(0, 0) / (con_matrix(0, 0) + con_matrix(1, 0));
}

double LogisticRegression::recall(const Dataset &X_test, const Dataset &y_test) const
{
    Eigen::MatrixXd con_matrix;
    confusion_matrix(X_test, y_test, con_matrix);
    return con_matrix(0, 0) / (con_matrix(0, 0) + con_matrix(0, 1));
}

double LogisticRegression::f1_score(const Dataset &X_test, const Dataset &y_test) const
{
    double precision_score = precision(X_test, y_test);
    double recall_score = recall(X_test, y_test);
    return 2 * (precision_score * recall_score) / (precision_score + recall_score);
}
=======
double LogisticRegression::sigmoid( double x) const{
    return 1 / (1 + std::exp(-x));
}

Eigen::VectorXd LogisticRegression::gradient(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(X.cols());
    for (int i = 0; i < X.rows(); i++) {
        grad += (y(i) - sigmoid(X.row(i).dot(*m_beta))) * X.row(i);
    }
    return grad.transpose();

}

double LogisticRegression::calculateAccuracy(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    int correctPredictions = 0;
    for (int i = 0; i < X.size(); i++) {
        Eigen::VectorXd instance = X.row(i);
        double prediction = estimate(instance);
        double actual = y.row(i)(0);  // assuming y is a Dataset of Eigen::VectorXd with one element
        if ((prediction >= 0.5 && actual == 1.0) || (prediction < 0.5 && actual == 0.0)) {
            correctPredictions++;
        }
    }
    return static_cast<double>(correctPredictions) / X.size();
}

double LogisticRegression::calculate_test_Accuracy(Dataset* X, Dataset* y) {
    int correctPredictions = 0;
    Eigen::MatrixXd X_test = construct_matrix();
    Eigen::VectorXd y_test = construct_y();
    std::cout << "Xtest rows then columns " << X_test.rows()<< "  "<< X_test.cols() << std::endl;
    std::cout << "ytest rows then columns " << y_test.rows()<< "  "<< y_test.cols() << std::endl;
    // std::cout << "Actual: " << y_test << std::endl;
    for (int i = 0; i < X_test.rows(); i++) {
        Eigen::VectorXd instance = X_test.row(i);
        double prediction = sigmoid(instance.transpose() * (*m_beta));
        // std::cout << "Prediction: " << prediction << std::endl;

        double actual = y_test.row(i)(0);  // assuming y is a Dataset of Eigen::VectorXd with one element
        if ((prediction >= 0.5 && actual == 1.0) || (prediction < 0.5 && actual == 0.0)) {
            correctPredictions++;
        }
    }
    return (correctPredictions) / (double)X_test.rows();
}


>>>>>>> origin/main
