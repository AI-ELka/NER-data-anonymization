#include<iostream>
#include<cassert>
#include "../Dataset/Dataset.hpp"
#include "LogisticRegression.hpp"
#include "../Dataset/Dataset.hpp"
#include "Regression.hpp"

LogisticRegression::LogisticRegression(Dataset* X, Dataset* y, double learning_rate, long epochs) : Regression(X, y) {
	m_beta = NULL;
	this->learning_rate = learning_rate;
	this->epochs = epochs;
    std::cout << "Setting coeffs" << std::endl;
	set_coefficients();
}

LogisticRegression::~LogisticRegression() {
    if (m_beta != nullptr) {
        delete m_beta;
        m_beta = nullptr;
    }
}

Eigen::MatrixXd LogisticRegression::construct_matrix() {
    Eigen::MatrixXd Xones(get_X()->get_nbr_samples(), get_X()->get_dim() + 1);
    Xones.col(0).setOnes();
    for (int i = 0; i < get_X()->get_nbr_samples(); i++) {
        for (int j = 0; j < get_X()->get_dim(); j++) {
            Xones(i, j + 1) = get_X()->get_instance(i)[j];
        }
    }
    return Xones;
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
        return nullptr;
    }
    return m_beta;
}

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
    double S = sigmoid((*m_beta)(0) + x.transpose() * m_beta->tail(m_beta->size() - 1));
    return S;
}

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
