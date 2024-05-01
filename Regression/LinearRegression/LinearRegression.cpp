#include<iostream>
#include<cassert>
#include "LinearRegression.hpp"
#include "Dataset.hpp"
#include "Regression.hpp"

LinearRegression::LinearRegression(Dataset* dataset, int col_regr) 
: Regression(dataset, col_regr) {
	m_beta = NULL;
	set_coefficients();
}

LinearRegression::~LinearRegression() {
	if (m_beta != NULL) {
		m_beta->resize(0);
		delete m_beta;
	}
}

Eigen::MatrixXd LinearRegression::construct_matrix() {
	long n= m_dataset->get_nbr_samples();
	long d= m_dataset->get_dim() ;

	Eigen::MatrixXd X(n, d);
	X.col(0).setOnes();
	for (int i=0; i<n;i++){
		const std::vector<double>& rowi = m_dataset->get_instance(i);
		int r=1;
		for (int j=0; j<d; j++){
			if (j==m_col_regr){
				continue;
			}
			else{
				X(i,r) = rowi[j];
				r++;
			}
		}

	}
	return X;
}

Eigen::VectorXd LinearRegression::construct_y() {
	long n= m_dataset->get_nbr_samples();
	Eigen::VectorXd y(n);
	for(int i=0;i<n;i++){
		const std::vector<double>& rowi = m_dataset->get_instance(i);
		y(i)=rowi[m_col_regr];
	}
	return y;
}

void LinearRegression::set_coefficients() {
	long d= m_dataset->get_dim() ;
	m_beta = new Eigen::VectorXd(d);
	Eigen::MatrixXd X=construct_matrix();
	Eigen::VectorXd y=construct_y();
	Eigen::MatrixXd A = X.transpose() * X;
	Eigen::VectorXd b = A.llt().solve(X.transpose()*y);
	for(int i=0; i<d; i++){
		(*m_beta)(i)=b(i);
	}

}

const Eigen::VectorXd* LinearRegression::get_coefficients() const {
	if (!m_beta) {
		std::cout <<"Coefficients have not been allocated." <<std::endl;
		return NULL;
	}
	return m_beta;
}

void LinearRegression::show_coefficients() const {
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

void LinearRegression::print_raw_coefficients() const {
	std::cout<< "{ ";
	for (int i = 0; i < m_beta->size() - 1; i++) {
		std::cout << (*m_beta)[i] << ", ";
	}
	std::cout << (*m_beta)[m_beta->size() - 1];
	std::cout << " }" << std::endl;
}

void LinearRegression::sum_of_squares(Dataset* dataset, double& ess, double& rss, double& tss) const {
	assert(dataset->get_dim()==m_dataset->get_dim());
	long n= dataset->get_nbr_samples();
	long d= m_dataset->get_dim() ;

	Eigen::MatrixXd X(n, d);
	X.col(0).setOnes();
	for (int i=0; i<n;i++){
		const std::vector<double>& rowi = dataset->get_instance(i);
		int r=1;
		for (int j=0; j<d; j++){
			if (j==m_col_regr){
				continue;
			}
			else{
				X(i,r) = rowi[j];
				r++;
			}
		}

	}

	Eigen::VectorXd y(n);
	for(int i=0;i<n;i++){
		const std::vector<double>& rowi = dataset->get_instance(i);
		y(i)=rowi[m_col_regr];
	}
	ess = 0;
	rss = 0;
	tss = 0;
	Eigen::VectorXd yhat=X * (*m_beta);
	double meany=y.mean();
	tss=(y.array()-meany).square().sum();
	rss=(y-yhat).squaredNorm();
    ess = (yhat.array()-meany).square().sum();

}

double LinearRegression::estimate(const Eigen::VectorXd & x) const {
	return (*m_beta)(0) + x.transpose() * (*m_beta).tail(x.size());
}
