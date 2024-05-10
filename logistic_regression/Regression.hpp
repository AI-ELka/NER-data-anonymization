#ifndef REGRESSION_HPP
#define REGRESSION_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include "../Dataset/Dataset.hpp"


class Regression{
    protected:
        Dataset* m_X;
        Dataset* m_y;
    public:
        Regression(Dataset* X, Dataset* y);
        /**
          The estimate method is virtual: it will depend on the Regression being of class Linear or Knn.
        */
        //virtual double estimate(const Eigen::VectorXd & x) const = 0;
        /**
          The getter for m_col_regr
        */
        Dataset* get_y() const;
          /**
            The getter for m_dataset
          */
        Dataset* get_X() const;
  };

  #endif //REGRESSION_HPP
