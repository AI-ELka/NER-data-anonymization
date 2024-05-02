#include<iostream>
#include<cassert>
#include "../Dataset/Dataset.hpp"
#include "MulticlassClassifier.hpp"
#include "../Dataset/Dataset.hpp"
#include "Regression.hpp"

MulticlassClassifier::MulticlassClassifier(Dataset* X, Dataset* y, double lr, long n_epoch) : Regression(X, y) {
	m_beta = NULL;
	learning_rate = lr;
	epochs = n_epoch;
	set_coefficients();
}

MulticlassClassifier::~MulticlassClassifier() {
    if (m_beta != nullptr) {
        delete m_beta;
        m_beta = nullptr;
    }
}

Eigen::MatrixXd MulticlassClassifier::construct_matrix() {
    const int num_samples = get_X()->get_nbr_samples();
    const int dim = get_X()->get_dim();

    Eigen::MatrixXd Xones(num_samples, dim + 1);
    Xones.col(0).setOnes();

    for (int i = 0; i < num_samples; ++i) {
        const std::vector<double>& instance = get_X()->get_instance(i);
        for (int j = 0; j < dim; ++j) {
            Xones(i, j + 1) = instance[j];
        }
    }
    return Xones;
}

Eigen::VectorXd MulticlassClassifier::construct_y(int j) {
    assert (j < get_y()->get_dim());
    
    const int num_samples = get_y()->get_nbr_samples();
    Eigen::VectorXd y_labels(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        y_labels(i) = get_y()->get_instance(i)[j];
    }
    return y_labels;
}

void MulticlassClassifier::set_coefficients() {
    Eigen::MatrixXd X_data = construct_matrix();
    m_beta = new Eigen::MatrixXd(X_data.cols(), get_y()->get_dim());
    
    for (int i = 0; i < get_y()->get_dim(); i++) {
        Eigen::VectorXd y_labels = construct_y(i);
        m_beta->col(i).setZero();
        for (int j = 0; j < epochs; j++) {
            m_beta->col(i) -= learning_rate * gradient(X_data, y_labels);
        }
    }
}

// const Eigen::VectorXd* LogisticRegression::get_coefficients() const {
//     if (!m_beta) {
//         std::cout << "Coefficients have not been allocated." << std::endl;
//         return nullptr;
//     }
//     return m_beta;
// }

// void LogisticRegression::show_coefficients() const {
//     if (!m_beta) {
//         std::cout << "Coefficients have not been allocated." << std::endl;
//         return;
//     }

//     if (m_beta->size() != m_X->get_dim() + 1) {
//         std::cout << "Warning, unexpected size of coefficients vector: " << m_beta->size() << std::endl;
//     }

//     std::cout << "beta = (";
//     for (int i = 0; i < m_beta->size(); i++) {
//         std::cout << " " << (*m_beta)[i];
//     }
//     std::cout << " )" << std::endl;
// }

// void LogisticRegression::print_raw_coefficients() const {
//     std::cout << "{ ";
//     for (int i = 0; i < m_beta->size() - 1; i++) {
//         std::cout << (*m_beta)[i] << ", ";
//     }
//     std::cout << (*m_beta)[m_beta->size() - 1];
//     std::cout << " }" << std::endl;
// }

Eigen::VectorXd LogisticRegression::estimate(const Eigen::VectorXd & x) const {
    Eigen::VectorXd probas = Eigen::VectorXd(get_y()->get_dim());
    for (int i = 0; i < get_y()->get_dim(); i++) {
        probas(i) = sigmoid((*m_beta).col(i)(0) + x.transpose() * m_beta->col(i).tail(m_beta->col(i).size() - 1));
    }
    return probas;
}

double LogisticRegression::sigmoid( double x) const{
    return 1 / (1 + std::exp(-x));
}

Eigen::VectorXd LogisticRegression::gradient(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(X.cols());
    for (int i = 0; i < X.rows(); i++) {
        grad += ( y(i) - sigmoid(X.row(i).dot(*m_beta)) ) * X.row(i);
    }
    return grad.transpose();
}
