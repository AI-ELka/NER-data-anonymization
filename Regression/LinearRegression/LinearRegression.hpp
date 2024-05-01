#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include <Eigen/Dense>
#include <Eigen/Core>
#include "Dataset.hpp"
#include "Regression.hpp"

/**
  The LinearRegression class inherits from the Regression class, stores the coefficient and provides a bunch of specific methods.
*/
class LinearRegression : public Regression {
private:
    /**
      The linear Regression coefficient.
    */
	Eigen::VectorXd* m_beta;
public:
    /**
      The linear Regression method fits a linear Regression coefficient to col_regr using the provided dataset. It calls set_coefficients under the hood.
     @param dataset a pointer to a dataset
     @param m_col_regr the integer of the column index of Y
    */
	LinearRegression(Dataset* dataset, int col_regr);
    /**
      The destructor (frees m_beta).
    */
  ~LinearRegression();

  /**
    A function to construct from the data the matrix X needed by LinearRegression.
  */
	Eigen::MatrixXd construct_matrix();

  /**
    A function to construct the vector y needed by LinearRegression.
  */
  Eigen::VectorXd construct_y() ;

  /**
    The setter method of the private attribute m_beta which is called by LinearRegression.
    It should use the functions construct_matrix and construct_y.
  */
	void set_coefficients();
  
  /**
      The getter method of the private attribute m_beta.
    */
	const Eigen::VectorXd* get_coefficients() const;
  
    /**
      Prints the contents of the private attribute m_beta.
    */
	void show_coefficients() const;
  /**
      Prints the contents of the private attribute m_beta in a line.
    */
	void print_raw_coefficients() const;
    /**
      The sum_of_squares method calculates the ESS, RSS and TSS that will be initialized, passed by reference and thereafter printed by test_linear.
    */
	void sum_of_squares(Dataset* dataset, double& ess, double& rss, double& tss) const;
    /**
      The estimate method outputs the predicted Y for a given point x.
     @param x the point for which to estimate Y.
    */
	double estimate(const Eigen::VectorXd & x) const;
};

#endif //LINEAR_REGRESSION_HPP
