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

Dataset::Dataset(const char* file, const bool process_y) {
    m_nsamples = 0;
    m_dim = -1;

    std::ifstream fin(file);
    
    if (fin.fail()) {
        std::cout << "Cannot read from file " << file << "!" << std::endl;
        exit(1);
    }
    
    std::string line;

    // Read the file line by line
    while (getline(fin, line) && m_nsamples < 10) {
        std::vector<double> row;
        std::stringstream s(line);
        
        int ncols = 0;
        std::string word;

        // Parse each line by comma
        while (getline(s, word, ',')) {
            if (process_y) {
                if (word == "I-PER") {
                    row.push_back(1);
                    ncols++;
                } else {
                    row.push_back(0);
                    ncols++;
                
                }
            } else {
                // Convert the string to double and add it to the row vector
                double val = std::stod(word); // Use std::stod for conversion
                row.push_back(val);
                ncols++;
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