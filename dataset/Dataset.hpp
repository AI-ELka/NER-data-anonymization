#ifndef DATASET_HPP
#define DATASET_HPP

#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>

class Dataset {
    public:

        Dataset(const char* file);

        ~Dataset();
            
        void show(bool verbose) const;

        const std::vector<double>& get_instance(int i) const;

    	int get_nbr_samples() const;

    	int get_dim() const;

    private:

		int m_dim;

		int m_nsamples;

        std::vector<std::vector<double> > m_instances;
};
#endif //DATASET_HPP