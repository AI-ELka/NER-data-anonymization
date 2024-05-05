#include<iostream>
#include<cassert>
#include "../Dataset/Dataset.hpp"
#include "MulticlassClassifier.hpp"
#include "../Dataset/Dataset.hpp"
#include "Regression.hpp"
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Core>

using namespace std;

MulticlassClassifier::MulticlassClassifier(Dataset* X, Dataset* y, double lr, long n_epoch) : Regression(X, y) {
	m_beta = nullptr;
	learning_rate = lr;
	epochs = n_epoch;
    map_labels=y->get_labels();
	set_coefficients();
}//use this one 

MulticlassClassifier::MulticlassClassifier(Dataset* X, Dataset* y, double lr, long n_epoch, vector<string> labels) : Regression(X, y) {
	m_beta = nullptr;
	learning_rate = lr;
	epochs = n_epoch;
    for (size_t i = 0; i < labels.size(); i++) {
        map_labels[labels[i]] = i;
    }
    if (map_labels != y->get_labels()) {
        std::cerr << "map_labels and y labels are not equal" << std::endl;
        std::exit(EXIT_FAILURE);
    }
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
    assert ( j < get_y()->get_dim() );
    
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
            m_beta->col(i) += learning_rate * gradient( X_data, y_labels, m_beta->col(i) );
        }
    }
}

double MulticlassClassifier::sigmoid( double x) const{
    return 1 / (1 + std::exp(-x));
}

Eigen::VectorXd MulticlassClassifier::gradient(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, const Eigen::VectorXd &beta) const {
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(X.cols());
    for (int i = 0; i < X.rows(); i++) {
        grad += ( y(i) - sigmoid(X.row(i).dot(beta)) ) * X.row(i);
    }
    return grad.transpose();
}

Eigen::VectorXd MulticlassClassifier::estimate(const Eigen::VectorXd & x) const {
    Eigen::VectorXd probas = Eigen::VectorXd(get_y()->get_dim());
    for (int i = 0; i < get_y()->get_dim(); i++) {
        probas(i) = sigmoid( (*m_beta).col(i)(0) + x.transpose() * m_beta->col(i).tail(m_beta->col(i).size() - 1) );
    }
    return probas;
}

const Eigen::MatrixXd* MulticlassClassifier::get_coefficients() const {
    if (!m_beta) {
        std::cout << "Coefficients have not been allocated." << std::endl;
        return nullptr;
    }
    return m_beta;
}

void MulticlassClassifier::confusion_matrix(const Dataset &X, const Dataset &y, Eigen::MatrixXd &con_matrix) const {
    con_matrix = Eigen::MatrixXd::Zero(y.get_dim(), y.get_dim());
    
    for (int i = 0; i < X.get_nbr_samples(); i++) {
        
        Eigen::VectorXd votes = Eigen::VectorXd::Zero(get_y()->get_dim());

        const std::vector<double> instance = X.get_instance(i);
        
        Eigen::VectorXd vec(instance.size());
        for (size_t i = 0; i < instance.size(); ++i) {
            vec(i) = instance[i];
        }
        Eigen::VectorXd probas = estimate(vec);

        for (int i = 0; i < probas.size(); i++) {
            if ( probas(i) >= 0.5 ) {
                votes(i) += 1;
            } else {
                for (int j = 0; j < probas.size(); j++) {
                    if (j != i) {
                        votes(j) += 1;
                    }
                }
            }
        }
        
        int prediction = 0;
        for (int i = 1; i < votes.size(); i++) {
            if (votes(i) > votes(prediction)) {
                prediction = i;
            }
        }

        for (int k = 0; k < y.get_dim(); k++) {
            if ( y.get_instance(i)[k] == 1 ) {
                con_matrix(k, prediction) += 1;
            }
        }
    }
}

double MulticlassClassifier::accuracy(const Dataset &X, const Dataset &y) const {
    Eigen::MatrixXd con_matrix;
    confusion_matrix(X, y, con_matrix);
    double accuracy = 0;
    for (int i = 0; i < con_matrix.rows(); i++) {
        accuracy += con_matrix(i, i);
    }
    accuracy /= (double)X.get_nbr_samples();
    return accuracy;
}

double MulticlassClassifier::precision(const Dataset &X, const Dataset &y) const {
    Eigen::MatrixXd con_matrix;
    confusion_matrix(X, y, con_matrix);
    double precision = 0;
    for (int i = 0; i < con_matrix.rows(); i++) {
        double true_positives = con_matrix(i, i);
        double false_positives = 0;
        for (int j = 0; j < con_matrix.rows(); j++) {
            if (j != i) {
                false_positives += con_matrix(j, i);
            }
        }
        precision += true_positives / (true_positives + false_positives);
    }
    precision /= (double)con_matrix.rows();
    return precision;
}

double MulticlassClassifier::recall(const Dataset &X, const Dataset &y) const {
    Eigen::MatrixXd con_matrix;
    confusion_matrix(X, y, con_matrix);
    double recall = 0;
    for (int i = 0; i < con_matrix.rows(); i++) {
        double true_positives = con_matrix(i, i);
        double false_negatives = 0;
        for (int j = 0; j < con_matrix.rows(); j++) {
            if (j != i) {
                false_negatives += con_matrix(i, j);
            }
        }
        recall += true_positives / (true_positives + false_negatives);
    }
    recall /= (double)con_matrix.rows();
    return recall;
}

double MulticlassClassifier::f1_score(const Dataset &X, const Dataset &y) const {
    double precision_score = precision(X, y);
    double recall_score = recall(X, y);
    return 2 * (precision_score * recall_score) / (precision_score + recall_score);
}