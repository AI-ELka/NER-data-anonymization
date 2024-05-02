<<<<<<< HEAD
#include <iostream>
#include <cassert>
=======
#include<iostream>
#include<cassert>
#include "../Dataset/Dataset.hpp"
>>>>>>> 3d134ca (add cmake)
#include "LogisticRegression.hpp"
#include "../Dataset/Dataset.hpp"
#include "Regression.hpp"

LogisticRegression::LogisticRegression(Dataset* X, Dataset* y, double learning_rate, long epochs) : Regression(X, y) {
<<<<<<< HEAD
    m_beta = nullptr;
    this->learning_rate = learning_rate;
    this->epochs = epochs;
    set_coefficients();
=======
	m_beta = NULL;
	this->learning_rate = learning_rate;
	this->epochs = epochs;
	set_coefficients();
>>>>>>> 3d134ca (add cmake)
}

LogisticRegression::~LogisticRegression() {
    if (m_beta != nullptr) {
        delete m_beta;
        m_beta = nullptr;
    }
}

Eigen::MatrixXd LogisticRegression::construct_matrix() {
    Eigen::MatrixXd X(get_X()->get_nbr_samples(), get_X()->get_dim() + 1);
    X.col(0).setOnes();

    for (int i = 0; i < get_X()->get_nbr_samples(); i++) {
        for (int j = 0; j < get_X()->get_dim(); j++) {
            X(i, j + 1) = get_X()->get_instance(i)[j];
        }
    }
    return X;
}

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

    m_beta = new Eigen::VectorXd(X.cols());
    m_beta->setZero();
    for (int i = 0; i < epochs; i++) {
        *m_beta -= learning_rate * gradient(X, y);
    }
}

const Eigen::VectorXd* LogisticRegression::get_coefficients() const {
    if (!m_beta) {
        std::cout << "Coefficients have not been allocated." << std::endl;
        return nullptr;
    }
    return m_beta;
}

void LogisticRegression::show_coefficients() const {
    if (!m_beta) {
        std::cout << "Coefficients have not been allocated." << std::endl;
        return;
    }

    if (m_beta->size() != X->get_dim() + 1) {
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
    double S = sigmoid((*m_beta)(0) + x.transpose() * m_beta->tail(m_beta->size() - 1));
    return S;
}

<<<<<<< HEAD
double LogisticRegression::sigmoid(const double x) const {
=======
double LogisticRegression::sigmoid( double x) const{
>>>>>>> 3d134ca (add cmake)
    return 1 / (1 + std::exp(-x));
}

Eigen::VectorXd LogisticRegression::gradient(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(X.cols());
    for (int i = 0; i < X.rows(); i++) {
        grad += (y(i) - sigmoid(X.row(i).dot(*m_beta))) * X.row(i);
    }
    return grad.transpose();
}
