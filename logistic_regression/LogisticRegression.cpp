#include <iostream>
#include <cassert>
#include "../Dataset/Dataset.hpp"
#include "LogisticRegression.hpp"
#include "../Dataset/Dataset.hpp"
#include "Regression.hpp"

using namespace std;

LogisticRegression::LogisticRegression(Dataset *X, Dataset *y, double lr, long m_epochs) : Regression(X, y)
{
    m_beta = nullptr;
    learning_rate = lr;
    epochs = m_epochs;
    set_coefficients();
}

LogisticRegression::~LogisticRegression()
{
    if (m_beta != nullptr)
    {
        delete m_beta;
        m_beta = nullptr;
    }
}

Eigen::MatrixXd LogisticRegression::construct_matrix()
{
    Eigen::MatrixXd Xones(get_X()->get_nbr_samples(), get_X()->get_dim() + 1);
    Xones.col(0).setOnes();
    for (int i = 0; i < get_X()->get_nbr_samples(); i++)
    {
        for (int j = 0; j < get_X()->get_dim(); j++)
        {
            Xones(i, j + 1) = get_X()->get_instance(i)[j];
        }
    }
    return Xones;
}

Eigen::VectorXd LogisticRegression::construct_y()
{
    Eigen::VectorXd y(get_y()->get_nbr_samples());
    for (int i = 0; i < get_y()->get_nbr_samples(); i++)
    {
        y(i) = get_y()->get_instance(i)[0];
    }
    return y;
}

void LogisticRegression::set_coefficients()
{
    Eigen::MatrixXd X = construct_matrix();
    Eigen::VectorXd y = construct_y();

    m_beta = new Eigen::VectorXd(X.cols());
    m_beta->setZero();
    for (int i = 0; i < epochs; i++)
    {
        *m_beta += learning_rate * gradient(X, y, *m_beta);
    }
}

const Eigen::VectorXd *LogisticRegression::get_coefficients() const
{
    if (!m_beta)
    {
        cout << "Coefficients have not been allocated." << endl;
        return nullptr;
    }
    return m_beta;
}

double LogisticRegression::estimate(const Eigen::VectorXd &x) const
{
    double S = sigmoid((*m_beta)(0) + x.transpose() * m_beta->tail(m_beta->size() - 1));
    return S;
}

double LogisticRegression::sigmoid(double x) const
{
    return 1 / (1 + exp(-x));
}

Eigen::VectorXd LogisticRegression::gradient(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, Eigen::VectorXd &beta) const
{
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(X.cols());
    for (int i = 0; i < X.rows(); i++)
    {
        grad += (y(i) - sigmoid(X.row(i).dot(beta))) * X.row(i);
    }
    return grad.transpose();
}

void LogisticRegression::confusion_matrix(const Dataset &X_test, const Dataset &y_test, Eigen::MatrixXd &con_matrix) const
{
    con_matrix = Eigen::MatrixXd::Zero(2, 2);
    for (int i = 0; i < X_test.get_nbr_samples(); i++)
    {
        const std::vector<double> instance = X_test.get_instance(i);
        Eigen::VectorXd vec(instance.size());
        for (size_t i = 0; i < instance.size(); ++i)
        {
            vec(i) = instance[i];
        }
        double prediction = estimate(vec);

        if (prediction >= 0.5 && y_test.get_instance(i)[0] == 1)
        {
            con_matrix(0, 0)++;
        }
        else if (prediction < 0.5 && y_test.get_instance(i)[0] == 0)
        {
            con_matrix(1, 1)++;
        }
        else if (prediction >= 0.5 && y_test.get_instance(i)[0] == 0)
        {
            con_matrix(1, 0)++;
        }
        else if (prediction < 0.5 && y_test.get_instance(i)[0] == 1)
        {
            con_matrix(0, 1)++;
        }
    }
}

double LogisticRegression::accuracy(const Dataset &X_test, const Dataset &y_test) const
{
    Eigen::MatrixXd con_matrix;
    confusion_matrix(X_test, y_test, con_matrix);
    return (con_matrix(0, 0) + con_matrix(1, 1)) / (con_matrix.sum());
}

double LogisticRegression::precision(const Dataset &X_test, const Dataset &y_test) const
{
    Eigen::MatrixXd con_matrix;
    confusion_matrix(X_test, y_test, con_matrix);
    return con_matrix(0, 0) / (con_matrix(0, 0) + con_matrix(1, 0));
}

double LogisticRegression::recall(const Dataset &X_test, const Dataset &y_test) const
{
    Eigen::MatrixXd con_matrix;
    confusion_matrix(X_test, y_test, con_matrix);
    return con_matrix(0, 0) / (con_matrix(0, 0) + con_matrix(0, 1));
}

double LogisticRegression::f1_score(const Dataset &X_test, const Dataset &y_test) const
{
    double precision_score = precision(X_test, y_test);
    double recall_score = recall(X_test, y_test);
    return 2 * (precision_score * recall_score) / (precision_score + recall_score);
}