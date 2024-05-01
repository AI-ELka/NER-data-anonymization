#include <iostream>
#include <ANN/ANN.h>
#include "KnnRegression.hpp"

KnnRegression::KnnRegression(int k, Dataset* dataset, int col_regr)
: Regression(dataset, col_regr) {
	m_k = k;
    int dim=dataset->get_dim();
    int npts = dataset->get_nbr_samples();
    m_dataPts= annAllocPts(npts , dim-1);
    for(int i=0 ; i<npts ; i++){
        const std::vector<double>& rowi=dataset->get_instance(i);
		std::cout<<rowi[dim-1]<<std::endl;

        int j=0, idx=0;
        while(idx<dim ){
            if(j==col_regr){
                idx++;
				continue;
            }
            m_dataPts[i][j]=rowi[idx];
            j++;idx++;
        }  

    }
    m_kdTree= new ANNkd_tree(m_dataPts ,npts , dim -1) ;
}

KnnRegression::~KnnRegression() {
    delete m_kdTree;
    annDeallocPts(m_dataPts);
}

double KnnRegression::estimate(const Eigen::VectorXd & x) const {
	assert(x.size()==m_dataset->get_dim()-1);
    ANNidxArray nnIdx= new ANNidx[m_k]; 
    ANNdistArray dists= new ANNdist[m_k];  //(squared)
	ANNpoint qPt = annAllocPt(m_dataset->get_dim()-1);
	for (int i = 0; i < m_dataset->get_dim()-1; i++) {
		qPt[i] = x[i];
	}
    double errorbound=0;
    m_kdTree->annkSearch(
        qPt,m_k,nnIdx,dists,errorbound
    );
    double sum=0;
    int output=0;
    for(int i=0; i<m_k ; i++){
        sum+=m_dataset->get_instance(nnIdx[i])[m_col_regr];
    }
    sum/=m_k;



	return sum;
}

int KnnRegression::get_k() const {
	return m_k;
}

ANNkd_tree* KnnRegression::get_kdTree() const {
	return m_kdTree;
}
