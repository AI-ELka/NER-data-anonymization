cmake_minimum_required(VERSION 3.10)
project(NER_Project)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the include directory for Eigen
include_directories(
    /home/mach/Desktop/eigen3/eigen-3.4.0  # Eigen library path
    ${CMAKE_CURRENT_SOURCE_DIR}/Dataset    # Your Dataset headers
    ${CMAKE_CURRENT_SOURCE_DIR}/logistic_regression  # Logistic Regression headers
)

# Add executable
add_executable(NER_Project main.cpp
    Dataset/Dataset.cpp
    logistic_regression/LogisticRegression.cpp
    logistic_regression/Regression.cpp
    logistic_regression/MulticlassClassifier.cpp
)
# If you have other directories with header files, include them as well
target_include_directories(NER_Project PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

