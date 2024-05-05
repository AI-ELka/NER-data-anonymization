#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include "Dataset.hpp"

int Dataset::get_nbr_samples() const {
	return m_nsamples;
}

int Dataset::get_dim() const {
	return m_dim;
}

std::map<std::string, int> Dataset::get_labels() const {
    return labelIndexMap;
}

Dataset::~Dataset() {
}

void Dataset::show(bool verbose) const {
	std::cout << "dataset with " << m_nsamples <<
		" samples, and " << m_dim << " dimensions." <<std::endl;

	if (verbose) {
		for (int i=0; i<m_nsamples; i++) {
			for (int j=0; j<m_dim; j++) {
				std::cout<<m_instances[i][j]<<" ";
			}
			std::cout<<std::endl;		
		}
	}
}


Dataset::Dataset(const char* file, const bool process_y, const bool multiclass, const std::optional<std::vector<std::string>>& labels) {
    if(process_y == multiclass) {
        std::cerr << "ERROR: process_y and multiclass cannot be the same \n Choose true flase (and then optional labels) if you want binary classification .\n For a mutliclass , choose false true and give the labels you gave in the dataset.\n Thank you!!" << std::endl;
        exit(-1);
    }
    if (labels.has_value()) {
        std::cout << "Labels provided" << std::endl;
        for (size_t i = 0; i < labels.value().size(); i++) {
            labelIndexMap[labels.value()[i]] = i;
        }
    }
    m_nsamples = 0;
    m_dim = -1;

    std::ifstream fin(file);
    
    if (fin.fail()) {
        std::cout << "Cannot read from file " << file << "!" << std::endl;
        exit(1);
    }
    
    std::string line;

    // Read the file line by line
    while ( getline(fin, line) ) {
        std::vector<double> row;
        std::stringstream s(line);
        
        int ncols = 0;
        std::string word;

        // Parse each line by comma
        while (getline(s, word, ',')) {
            if (multiclass) {
                row=encodeLabel(word);
                ncols += row.size();
            } else {
                if (process_y) {
                    if (word == "I-PER") {
                        row.push_back(1);
                        ncols++;
                    } else {
                        row.push_back(0);
                        ncols++;
                    }
                } else {
                    double val = std::stod(word);
                    row.push_back(val);
                    ncols++;
                }
            }
        }

        // Check if any columns were read
        if (ncols == 0)
            continue; // Skip empty lines

        // Store the row vector in the Dataset
        m_instances.push_back(row);

        // Update the dimension if it's not set yet
        if (m_dim == -1)
            m_dim = ncols;
        else if (m_dim != ncols) {
            std::cerr << "ERROR: Inconsistent dataset dimensions" << std::endl;
            exit(-1);
        }

        m_nsamples++;
    }
    
    fin.close();
    std::cout << "Loaded " << m_nsamples << " samples with " << m_dim << " dimensions." << std::endl;
}

const std::vector<double>& Dataset::get_instance(int i) const {
	return m_instances[i];
}

// std::vector<double> Dataset::encodeLabel(const std::string& label) {
//     if (label == "O") {
//         return {1, 0, 0, 0, 0, 0, 0, 0, 0}; //one-hot encoding
//     } else if (label == "B-PER") {
//         return {0, 1, 0, 0, 0, 0, 0, 0, 0}; 
//     } else if (label == "I-PER") {
//         return {0, 0, 1, 0, 0, 0, 0, 0, 0}; 
//     } else if (label == "B-ORG") {
//         return {0, 0, 0, 1, 0, 0, 0, 0, 0}; 
//     } else if (label == "I-ORG") {
//         return {0, 0, 0, 0, 1, 0, 0, 0, 0}; 
//     } else if (label == "B-LOC") {
//         return {0, 0, 0, 0, 0, 1, 0, 0, 0}; 
//     } else if (label == "I-LOC") {
//         return {0, 0, 0, 0, 0, 0, 1, 0, 0}; 
//     } else if (label == "B-MISC") {
//         return {0, 0, 0, 0, 0, 0, 0, 1, 0}; 
//     } else if (label == "I-MISC") {
//         return {0, 0, 0, 0, 0, 0, 0, 0, 1}; 
//     } else {
//         std::cout << label << std::endl;
//         std::cerr << "ERROR: Unknown class label " << label << std::endl;
//         exit(-1);
//     }
// }

std::vector<double> Dataset::encodeLabel(const std::string& label) {
    if (labelIndexMap.find(label) != labelIndexMap.end()) {
        std::vector<double> oneHot(labelIndexMap.size(), 0.0);
        oneHot[labelIndexMap[label]] = 1.0;  // Set the corresponding index to 1
        return oneHot;
    } else {
        std::cout << label << std::endl;
        std::cerr << "ERROR: Unknown class label " << label << std::endl;
        exit(-1);
    }
}
