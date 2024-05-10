#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include <Eigen/Dense>
#include <Eigen/Core>
#include "../Dataset/Dataset.hpp"
#include "Regression.hpp"

class LogisticRegression : public Regression
{
private:
  /**
    The logistic Regression vector of coefficients and learning hyperparameters.
  */
  Eigen::VectorXd *m_beta;
  double learning_rate;
  long epochs;

public:
  /**
    @param X a pointer to a dataset
    @param y the integer of the column index of Y
  */
  LogisticRegression(Dataset *X, Dataset *y, double lr, long m_epochs);
  
  /**
    The destructor (frees m_beta).
  */
  ~LogisticRegression();

  /**
    A function to construct from the data the matrix X needed by LogisticRegression.
  */
  Eigen::MatrixXd construct_matrix();

  /**
    A function to construct the vector y needed by LogisticRegression.
  */
  Eigen::VectorXd construct_y();

  /**
    The setter method of the private attribute m_beta which is called by LogisticRegression.
  */
  void set_coefficients();

  /**
    The getter method of the private attribute m_beta.
  */
  const Eigen::VectorXd *get_coefficients() const;
  
  /**
    The estimate method outputs the predicted y for a given point x.
    @param x the point for which to estimate y.
  */
  double estimate(const Eigen::VectorXd &x) const;

  /**
    The sigmoid method calculates the sigmoid of a given value.
    @param x the value for which to calculate the sigmoid.
  */
  double sigmoid(double x) const;

  /**
    The gradient method calculates the gradient of the maximum likelihood function.
    @param X the matrix of the dataset.
    @param y the vector of the labels.
    @param beta the vector of the coefficients.
  */
  Eigen::VectorXd gradient(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, Eigen::VectorXd &beta) const;

  /**
    The sum_of_squares method calculates the ESS, RSS and TSS that will be initialized, passed by reference and thereafter printed by test_linear.
    @param X the matrix of the dataset.
    @param y the vector of the labels.
    @param con_matrix the confusion matrix.
  */
  void confusion_matrix(const Dataset &X, const Dataset &y, Eigen::MatrixXd &con_matrix ) const;

  /**
    The accuracy method calculates the accuracy of the model on a given dataset.
    @param X the matrix of the features.
    @param y the vector of the labels.
  */
  double accuracy(const Dataset &X, const Dataset &y) const;

  /**
    The double precision method calculates the double precision of the model on a given dataset.
    @param X the matrix of the features.
    @param y the vector of the labels.
  */
  double precision(const Dataset &X, const Dataset &y) const;

  /**
    The double recall method calculates the double recall of the model on a given dataset.
    @param X the matrix of the features.
    @param y the vector of the labels.
  */
  double recall(const Dataset &X, const Dataset &y) const;

  /**
    The double f1_score method calculates the double f1_score of the model on a given dataset.
    @param X the matrix of the features.
    @param y the vector of the labels.
  */
  double f1_score(const Dataset &X, const Dataset &y) const;
};

#endif
