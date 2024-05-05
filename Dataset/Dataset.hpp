#ifndef DATASET_HPP
#define DATASET_HPP

#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <map>
#include <optional>
class Dataset {
    public:

        Dataset(const char* file, bool process_y = false, bool multiclass = false, const std::optional<std::vector<std::string>>& labels = std::nullopt);

        ~Dataset();
            
        void show(bool verbose) const;

        const std::vector<double>& get_instance(int i) const;

    	int get_nbr_samples() const;

    	int get_dim() const;

        std::map<std::string, int> get_labels() const;

        std::vector<double> encodeLabel(const std::string& label);

    private:
        std::map<std::string, int> labelIndexMap;
        
		int m_dim;

		int m_nsamples;

        std::vector<std::vector<double> > m_instances;
};
#endif //DATASET_HPP