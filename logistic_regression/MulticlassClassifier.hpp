#ifndef MULTICLASSCLASSIFIER_HPP
#define MULTICLASSCLASSIFIER_HPP

#include <Eigen/Dense>
#include <Eigen/Core>
#include "../Dataset/Dataset.hpp"
#include "Regression.hpp"
#include <map>
#include <vector>
#include <string>

using namespace std;

class MulticlassClassifier : public Regression
{
private:
  /*
    The multiclass classification coefficients and learning hyperparameters.
  */
  Eigen::MatrixXd *m_beta;
  double learning_rate;
  long epochs;

  /*
    The mapping of the labels of the classes.
  */
  map<string, int> map_labels;

public:
  /**
    @param X a pointer to features
    @param y a pointer to labels
    @param learning_rate the learning rate of the gradient descent
    @param epochs the number of epochs
    @param labels the labels of the classes
  */
    MulticlassClassifier(Dataset *X, Dataset *y, double lr, long n_epoch);
    MulticlassClassifier(Dataset *X, Dataset *y, double learning_rate, long epochs, vector<string> labels);

    /*
      The destructor (frees m_beta).
    */
    ~MulticlassClassifier();

    /*
      A function to construct from the data the matrix X needed by LogisticRegression.
    */
    Eigen::MatrixXd construct_matrix();

    /*
      A function to construct the vector y needed by LogisticRegression.
    */
    Eigen::VectorXd construct_y(int j);

    /*
      The setter method of the private attribute m_beta which is called by LogisticRegression.
    */
    void set_coefficients();

    /**
      The getter method of the private attribute m_beta.
    */
    const Eigen::MatrixXd *get_coefficients() const;

    /**
      The estimate method outputs the predicted probabilities for one vs all classes for a given point x.
      @param x the point for which to estimate Y.
    */
    Eigen::VectorXd estimate(const Eigen::VectorXd &x) const;

    /**
      The sigmoid method calculates the sigmoid of a given value.
      @param x the value for which to calculate the sigmoid.
    */
    double sigmoid(double x) const;

    /**
        The gradient method calculates the gradient of the loss function.
        @param X the matrix of the dataset.
        @param y the vector of the labels.
        @param beta the vector of the coefficients.
    */
    Eigen::VectorXd gradient(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, const Eigen::VectorXd &beta) const;

    /**
        Compute the confusion matrix of the model.
        @param X the matrix of the dataset.
        @param y the vector of the labels.
    */
    void confusion_matrix(const Dataset &X, const Dataset &y, Eigen::MatrixXd &con_matrix) const;

    /**
      The accuracy method calculates the accuracy of the model.
      @param X the matrix of the dataset.
      @param y the vector of the labels.
    */
    double accuracy(const Dataset &X, const Dataset &y) const;

    /**
      The precision method calculates the precision of the model.
      @param X the matrix of the dataset.
      @param y the vector of the labels.
    */
    double precision(const Dataset &X, const Dataset &y) const;

    /**
      The recall method calculates the recall of the model.
      @param X the matrix of the dataset.
      @param y the vector of the labels.
    */
    double recall(const Dataset &X, const Dataset &y) const;

    /**
      The f1_score method calculates the f1_score of the model.
      @param X the matrix of the dataset.
      @param y the vector of the labels.
    */
    double f1_score(const Dataset &X, const Dataset &y) const;
};

#endif
