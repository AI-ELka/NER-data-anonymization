<<<<<<< HEAD
# Named Entity Recognition Project

This project is a Named Entity Recognition (NER) system implemented in C++ using logistic regression and multiclass classification.

## Project Structure

The project is organized into several directories and files:

- `main.cpp`: This is the main entry point of the application. It handles user input to determine whether to perform binary or multiclass classification, loads the datasets, creates instances of the LogisticRegression or MulticlassClassifier classes, fits the models, and evaluates them. Change the data path to your desired train & test datasets. Find more dataset here : [juand-r /
entity-recognition-datasets](https://github.com/juand-r/entity-recognition-datasets/tree/master?tab=readme-ov-file), Preprocess them with the `useful/Anonymization.ipynb` .

- `logistic_regression/`: This directory contains the implementation of the logistic regression model and the multiclass classifier. It includes the following files:
  - `LogisticRegression.cpp` and `LogisticRegression.hpp`: These files define the LogisticRegression class, which implements a logistic regression model. The class includes methods for estimating the model, calculating the sigmoid function, and computing the gradient. It also includes methods for calculating various metrics, such as accuracy, precision, recall, and F1 score.
  - `MulticlassClassifier.cpp` and `MulticlassClassifier.hpp`: These files define the MulticlassClassifier class, which implements a multiclass classification model.

- `Dataset/`: This directory contains the implementation of the Dataset class, which is used to load and manage the datasets used in the project.

- `CMakeLists.txt`: This file is used by CMake to build the project. It specifies the required version of CMake, the C++ standard to use, the directories containing the header files, and the source files to compile.

- `data/`: This directory contains the datasets used in the project. The datasets are in CSV and NPY formats.

- `requirements.txt`: This file lists the dependencies required to run the project.


## How to Run the Project

To build and run the project, follow these steps:

1. Ensure that you have the required dependencies installed. You can install them using the command `pip install -r requirements.txt`.

2. Build the project using CMake. Navigate to the project directory and run the command `cmake .`.

3. After the project has been built, you can run it using the command `./my_program`.

When you run the program, you will be prompted to enter 'binary' for binary classification or 'multiclass' for multiclass classification. The program will then load the appropriate datasets, fit the model, and evaluate it.

## License

This project is licensed under the terms of the MIT license. See the `LICENSE` file for details.
=======
jfjfjffkf
>>>>>>> origin/main
