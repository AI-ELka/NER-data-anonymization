#include<iostream>
#include<cassert>
#include "LogisticRegression.hpp"
#include "Dataset.hpp"
#include "Regression.hpp"

LogisticRegression::LogisticRegression(Dataset* X, Dataset* y) : Regression(X, y) {
	m_beta = NULL;
	set_coefficients();
}

LogisticRegression::~LogisticRegression() {
	if (m_beta != NULL) {
		m_beta->resize(0);
		delete m_beta;
	}
}

Eigen::MatrixXd LogisticRegression::construct_matrix() {
    Eigen::MatrixXd X(get_X()->get_nbr_samples(), get_X()->get_dim()+1);
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
	m_beta = new Eigen::VectorXd((X.transpose() * X).inverse() * X.transpose() * y);

}

const Eigen::VectorXd* LogisticRegression::get_coefficients() const {
	if (!m_beta) {
		std::cout <<"Coefficients have not been allocated." <<std::endl;
		return NULL;
	}
	return m_beta;
}

void LogisticRegression::show_coefficients() const {
	if (!m_beta) {
		std::cout << "Coefficients have not been allocated." <<std::endl;
		return;
	}
	
	if (m_beta->size() != m_dataset->get_dim()) {  // ( beta_0 beta_1 ... beta_{d} )
		std::cout << "Warning, unexpected size of coefficients vector: " << m_beta->size() << std::endl;
	}
	
	std::cout<< "beta = (";
	for (int i=0; i<m_beta->size(); i++) {
		std::cout << " " << (*m_beta)[i];
	}
	std::cout << " )" <<std::endl;
}

void LogisticRegression::print_raw_coefficients() const {
	std::cout<< "{ ";
	for (int i = 0; i < m_beta->size() - 1; i++) {
		std::cout << (*m_beta)[i] << ", ";
	}
	std::cout << (*m_beta)[m_beta->size() - 1];
	std::cout << " }" << std::endl;
}

void LogisticRegression::sum_of_squares(Dataset* dataset, double& ess, double& rss, double& tss) const {
	assert(dataset->get_dim()==m_dataset->get_dim());
	// TODO Exercise 4

	ess = 0;
	rss = 0;
	tss = 0;
	for (int i = 0; i < dataset->get_nbr_samples(); i++) {
		double y = dataset->get_instance(i)[get_col_regr()];
		Eigen::VectorXd x(dataset->get_dim() - 1);
		for (int j = 0; j < dataset->get_dim(); j++) {
			if (j < get_col_regr()) {
				x(j) = dataset->get_instance(i)[j];
			} else if (j > get_col_regr()) {
				x(j - 1) = dataset->get_instance(i)[j];
			}
		}
		double y_hat = estimate(x);
		ess += (y_hat - y) * (y_hat - y);
		rss += (y_hat - y) * (y_hat - y);
		tss += (y - y_hat) * (y - y_hat);
	}

}

double LogisticRegression::estimate(const Eigen::VectorXd & x) const {
	double S = get_coefficients()->operator()(0);
	for (int i = 1; i < m_dataset->get_dim(); i++) {
		S += get_coefficients()->operator()(i) * x(i - 1);
	}
	return S;
}
