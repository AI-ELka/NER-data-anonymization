#ifndef MULTICLASSCLASSIFIER_HPP
#define MULTICLASSCLASSIFIER_HPP
#include <Eigen/Dense>
#include <Eigen/Core>
#include "../Dataset/Dataset.hpp"
#include "Regression.hpp"

class MulticlassClassifier : public Regression {
    private:
        /**
          The multiclass classification coefficients.
        */
        Eigen::matrixXd* m_beta;
        double learning_rate;
        long epochs;
    public:
        /**
        @param X a pointer to features
        @param y a pointer to labels
        */
        MulticlassClassifier(Dataset* X, Dataset* y, double learning_rate, long epochs);
        /**
          The destructor (frees m_beta).
        */
        ~MulticlassClassifier();

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
            It should use the functions construct_matrix and construct_y.
        */
        void set_coefficients();

        /**
          The getter method of the private attribute m_beta.
        */
        //const Eigen::VectorXd *get_coefficients() const;

        /**
          Prints the contents of the private attribute m_beta.
        */
        //void show_coefficients() const;
        /**
          Prints the contents of the private attribute m_beta in a line.
        */
        //void print_raw_coefficients() const;
        /**
          The sum_of_squares method calculates the ESS, RSS and TSS that will be initialized, passed by reference and thereafter printed by test_linear.
        */
        //void sum_of_squares(Dataset *dataset, double &ess, double &rss, double &tss) const;
        /**
            The estimate method outputs the predicted Y for a given point x.
            @param x the point for which to estimate Y.
        */
        Eigen::VectorXd estimate(const Eigen::VectorXd &x) const;

        // double sigmoid(const double x);

        double sigmoid(double x) const;

        /**
            The gradient method calculates the gradient of the loss function.
            @param X the matrix of the dataset.
            @param y the vector of the labels.
        */
        Eigen::VectorXd gradient(const Eigen::MatrixXd & X, const Eigen::VectorXd & y);
};

#endif