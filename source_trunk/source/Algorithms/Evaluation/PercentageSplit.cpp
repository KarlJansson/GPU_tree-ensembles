#include "stdafx.h"
#include "PercentageSplit.h"
#include "IAlgorithm.h"

namespace DataMiner{
	PercentageSplit::PercentageSplit(float splitPercentage):m_splitPercentage(splitPercentage){
		m_stage = 0;
		m_numStages = 1;
	}

	void PercentageSplit::run(IAlgorithmPtr algo, AlgorithmDataPackPtr data){
		setData(data->m_recMgr->getDocumentResource(data->m_dataResource),data);
		algo->start();
		algo->getoutputStream() <<
			"================================================================================\r\n";
		algo->getoutputStream() << "				Percentage Split: \r\n" <<
			"================================================================================\r\n\r\n";
		algo->runAlgorithm(data,shared_from_this());
		algo->getoutputStream() <<
			"\r\n================================================================================\r\n";

		data->m_gui->setText(IDC_STATIC_INFOTEXT,algo->getoutputStream().str());
		data->m_minerGUI->enableAllButStop();
	}

	bool PercentageSplit::advance(){
		if(m_stage == 0){
			m_stage++;
			return true;
		}
		return false;
	}

	bool PercentageSplit::isFinalStage(){
		if(m_stage == 1)
			return true;
		return false;
	}

	void PercentageSplit::init(){
		for(unsigned int i=0; i<m_data->getNumInstances(); i++){
			m_clInstances[m_data->getInstance(i)->classValue()].push_back(m_data->getInstance(i));
		}

		std::vector<unsigned int> numTrainClass(m_clInstances.size(),0);
		for(unsigned int i=0; i<numTrainClass.size(); ++i){
			numTrainClass[i] = m_clInstances[i].size() * (m_splitPercentage*0.01);
		}

		if(m_randomSelection){
			for(unsigned int i=0; i<m_clInstances.size(); ++i){
				std::vector<unsigned int> selectionVector;
				selectionVector.reserve(m_clInstances[i].size());
				for(unsigned int j=0; j<m_clInstances[i].size(); ++j){
					selectionVector.push_back(j);
				}

				unsigned int rngNr;
				boost::random::mt19937 rng;
				rng.seed(123);
				for(unsigned int j=0; j<m_clInstances[i].size(); ++j){
					boost::random::uniform_int_distribution<> indRand(0,m_clInstances[i].size()-1-i);
					rngNr = indRand(rng);
					if(j < numTrainClass[i])
						m_trainingInds.push_back(m_clInstances[i][selectionVector[rngNr]]->getIndex());
					else
						m_testingInds.push_back(m_clInstances[i][selectionVector[rngNr]]->getIndex());

					selectionVector[rngNr] = selectionVector[m_clInstances[i].size()-1-i];
				}
			}
		}
		else{
			for(unsigned int i=0; i<m_clInstances.size(); ++i){
				for(unsigned int j=0; j<m_clInstances[i].size(); j++){
					if(j < numTrainClass[i])
						m_trainingInds.push_back(m_clInstances[i][j]->getIndex());
					else
						m_testingInds.push_back(m_clInstances[i][j]->getIndex());
				}
			}
		}

		//scrambleInstances();
		calculateCost(numTrainClass[0],numTrainClass[1]);
	}
}