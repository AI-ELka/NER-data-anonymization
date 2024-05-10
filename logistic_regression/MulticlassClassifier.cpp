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

#include <chrono>

using namespace std;

MulticlassClassifier::MulticlassClassifier(Dataset* X, Dataset* y, double lr, long n_epoch, string md ) : Regression(X, y) {
	m_beta = nullptr;
	learning_rate = lr;
	epochs = n_epoch;
    mode = md;
    
    // Fit the model to the data and measure the time taken
    auto start = chrono::high_resolution_clock::now();

	set_coefficients();
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Time taken to fit the model: " << elapsed.count() << " s" << endl;
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
        const vector<double>& instance = get_X()->get_instance(i);
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
    if (mode == "one_vs_all") {
        Eigen::MatrixXd X_data = construct_matrix();
        m_beta = new Eigen::MatrixXd(X_data.cols(), get_y()->get_dim());
        
        for (int i = 0; i < get_y()->get_dim(); i++) {
            // Construct the vector y
            Eigen::VectorXd y_labels = construct_y(i);

            // Fit the model to the data and measure the time taken
            auto start2 = chrono::high_resolution_clock::now();

            m_beta->col(i).setZero();
            for (int j = 0; j < epochs; j++) {
                m_beta->col(i) += learning_rate * gradient( X_data, y_labels, m_beta->col(i) );
            }

            auto end2 = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed2 = end2 - start2;
            cout << "Time taken to fit the model for label " << i << ": " << elapsed2.count() << " s" << endl;
        }
    } else if (mode == "one_vs_one") {
        Eigen::MatrixXd X_data = construct_matrix();
        m_beta = new Eigen::MatrixXd(X_data.cols(), get_y()->get_dim() * (get_y()->get_dim() - 1) / 2);
        
        int k = 0;
        for (int i = 0; i < get_y()->get_dim()-1; i++) {
            for (int j = i + 1; j < get_y()->get_dim(); j++) {
                // Compute the number of samples for which the labels are i or j
                int num_samples = 0;
                for (int l = 0; l < get_X()->get_nbr_samples(); l++) {
                    if ( get_y()->get_instance(l)[i] == 1 || get_y()->get_instance(l)[j] == 1 ) {
                        num_samples++;
                    }
                }
                
                // Construct the matrix X_d and the vector y_labels
                Eigen::MatrixXd X_d(num_samples, X_data.cols());
                Eigen::VectorXd y_labels(num_samples);
                int m = 0;
                for (int l = 0; l < get_X()->get_nbr_samples(); l++) {
                    if ( get_y()->get_instance(l)[i] == 1 || get_y()->get_instance(l)[j] == 1 ) {
                        X_d.row(m) = X_data.row(l);
                        y_labels(m) = get_y()->get_instance(l)[i];
                        m++;
                    }
                }
                
                // Fit the model to the data and measure the time taken
                auto start2 = chrono::high_resolution_clock::now();

                m_beta->col(k).setZero();
                for (int l = 0; l < epochs; l++) {
                    m_beta->col(k) += learning_rate * gradient( X_d, y_labels, m_beta->col(k) );
                }

                auto end2 = chrono::high_resolution_clock::now();
                chrono::duration<double> elapsed2 = end2 - start2;
                cout << "Time taken to fit the model for labels " << i << " and " << j << ": " << elapsed2.count() << " s" << endl;
                k++;
            }
        }
    } else {
        cout << "Invalid mode." << endl;
    }
}
Eigen::VectorXd MulticlassClassifier::sigmoid(const Eigen::VectorXd& z) const {
    return 1.0 / (1.0 + (-z.array()).exp());
}
double MulticlassClassifier::sigmoid(double z) const {
    return 1.0 / (1.0 + exp(-z));
}

Eigen::VectorXd MulticlassClassifier::gradient(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, const Eigen::VectorXd &beta) const {
    Eigen::VectorXd sigmoid_values = sigmoid(X * beta);
    Eigen::VectorXd errors = y - sigmoid_values;

    Eigen::VectorXd grad = X.transpose() * errors;

    return grad;
}

Eigen::VectorXd MulticlassClassifier::estimate(const Eigen::VectorXd &x ) const {
    if (mode == "one_vs_all") {
        Eigen::VectorXd probas = Eigen::VectorXd(get_y()->get_dim());
        for (int i = 0; i < get_y()->get_dim(); i++) {
            probas(i) = sigmoid( (*m_beta).col(i)(0) + x.transpose() * m_beta->col(i).tail(m_beta->col(i).size() - 1) );
        }
        return probas;
    } else if (mode == "one_vs_one") {
        Eigen::VectorXd votes = Eigen::VectorXd::Zero(get_y()->get_dim());
        int k = 0;
        for (int i = 0; i < get_y()->get_dim()-1; i++) {
            for (int j = i + 1; j < get_y()->get_dim(); j++) {
                double proba = sigmoid( (*m_beta).col(k)(0) + x.transpose() * m_beta->col(k).tail(m_beta->col(k).size() - 1) );
                if (proba >= 0.5) {
                    votes(i) += 1.;
                } else {
                    votes(j) += 1.;
                }
                k++;
            }
        }
        return votes;
    } else {
        cout << "Invalid mode." << endl;
        return Eigen::VectorXd();
    }
}

const Eigen::MatrixXd* MulticlassClassifier::get_coefficients() const {
    if (!m_beta) {
        cout << "Coefficients have not been allocated." << endl;
        return nullptr;
    }
    return m_beta;
}

void MulticlassClassifier::confusion_matrix(const Dataset &X, const Dataset &y, Eigen::MatrixXd &con_matrix) const {
    if (mode == "one_vs_all") {
        con_matrix = Eigen::MatrixXd::Zero(y.get_dim(), y.get_dim());
        
        for (int i = 0; i < X.get_nbr_samples(); i++) {
            
            Eigen::VectorXd votes = Eigen::VectorXd::Zero(get_y()->get_dim());

            const vector<double> instance = X.get_instance(i);
            
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
    } else if (mode == "one_vs_one") {
        con_matrix = Eigen::MatrixXd::Zero(y.get_dim(), y.get_dim());
        
        for (int i = 0; i < X.get_nbr_samples(); i++) {
            
            const vector<double> instance = X.get_instance(i);
            
            Eigen::VectorXd vec(instance.size());
            for (size_t i = 0; i < instance.size(); ++i) {
                vec(i) = instance[i];
            }
            Eigen::VectorXd votes = estimate(vec);

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
    } else {
        cout << "Invalid mode." << endl;
    }
}

void MulticlassClassifier::show_confusion_matrix(const Dataset &X, const Dataset &y) const {
    Eigen::MatrixXd con_matrix;
    confusion_matrix(X, y, con_matrix);
    cout << "Confusion matrix:" << endl;
    cout << con_matrix << endl;
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
        if (true_positives + false_positives > 0) {
           precision += true_positives / (true_positives + false_positives);
        }
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
        if (true_positives + false_negatives > 0) {
            recall += true_positives / (true_positives + false_negatives);
        }
    }
    recall /= (double)con_matrix.rows();
    return recall;
}

double MulticlassClassifier::f1_score(const Dataset &X, const Dataset &y) const {
    double precision_score = precision(X, y);
    double recall_score = recall(X, y);
    return 2 * (precision_score * recall_score) / (precision_score + recall_score);
}