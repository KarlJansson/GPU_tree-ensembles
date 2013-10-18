#include "stdafx.h"
#include "NoEvaluation.h"

namespace DataMiner{
	void NoEvaluation::run(IAlgorithmPtr algo, AlgorithmDataPackPtr data){
			
	}
	
	bool NoEvaluation::advance(){
		return true;
	}
	
	void NoEvaluation::init(){
		for(unsigned int i=0; i<m_data->getNumInstances(); i++){
			m_clInstances[m_data->getInstance(i)->classValue()].push_back(m_data->getInstance(i));
			m_trainingInds.push_back(m_data->getInstance(i)->getIndex());
		}
	}
	
	bool NoEvaluation::isFinalStage(){
		return true;
	}
}