#include "stdafx.h"
#include "CrossValidation.h"
#include "GUIManager.h"
#include "IAlgorithm.h"

namespace DataMiner{
	CrossValidation::CrossValidation(unsigned int folds){
		m_useValidationSet = false;
		m_numStages = folds;
		m_stage = 0;
	}

	void CrossValidation::run(IAlgorithmPtr algo, AlgorithmDataPackPtr data){
		setData(data->m_recMgr->getDocumentResource(data->m_dataResource),data);

		algo->start();
		std::wstringstream &outStream = algo->getoutputStream();
		outStream << 
			"================================================================================\r\n";
		outStream << "				" << m_numStages << "-Fold Cross-Validation: \r\n" <<
			"================================================================================\r\n\r\n";
		while(advance()){
			if(m_stage <= m_numStages)
				outStream << "_Fold " << m_stage << ":_________________________________________________________________\r\n\r\n";
			else
				outStream << "_Validation set result:______________________________________________________\r\n\r\n";
			
			algo->runAlgorithm(data,shared_from_this());

			outStream << "\r\n\r\n";

			// Write summation for cross validation
			if(m_stage == m_numStages){
				IResultWriterPtr resPack = algo->getResultWriter();

				outStream << "_" << m_numStages << "-Fold Cross-Validation result:" <<
					"_______________________________________________\r\n\r\n";

				outStream <<
					"	Total time: " << resPack->getDouble("totalTime") << "s\r\n" <<
					"	Training time: " << resPack->getDouble("trainingTime") << "s\r\n" <<
					"	Testing time: " << resPack->getDouble("testingTime") << "s\r\n\r\n";

				outStream <<
					"	Total training amount: " << resPack->getDouble("trainingInstances") << " instances\r\n" <<
					"	Total testing amount: " << resPack->getDouble("testingInstances") << " instances";

				outStream << 
					"\r\n\r\n	" << 1 << "	" << 2 
					<< "\r\n	" << resPack->getDouble("cl1Correct") << "	" << resPack->getDouble("cl2Wrong") << "	" << 1 
					<< "\r\n	" << resPack->getDouble("cl1Wrong") << "	" << resPack->getDouble("cl2Correct") << "	" << 2 << "\r\n";

				outStream << 
					"\r\n	Accuracy: " << float(resPack->getDouble("accuracy"))/float(m_numStages) << "%" << "\r\n" <<
					"	AUC: " << resPack->getDouble("auc")/float(m_numStages) << "\r\n" <<
					"	Enrichment factor Cl1: " << resPack->getDouble("cl1EnrichmentFactor")/float(m_numStages) << "\r\n" <<
					"	Enrichment factor Cl2: " << resPack->getDouble("cl2EnrichmentFactor")/float(m_numStages) << "\r\n" <<
					"	CPU memory: " << resPack->getDouble("memUsageCPU")/float(m_numStages) << "\r\n" <<
					"	GPU memory: " << resPack->getDouble("memUsageGPU")/float(m_numStages) << "\r\n";

				outStream << "\r\n\r\n";

				std::stringstream fileMessage;
				fileMessage << float(resPack->getDouble("accuracy"))/float(m_numStages) << "	"
							<< resPack->getDouble("auc")/float(m_numStages) << "	" 
							<< resPack->getDouble("trainingTime")/float(m_numStages) << "	"
							<< resPack->getDouble("testingTime")/float(m_numStages) << "	"
							<< resPack->getDouble("memUsageCPU")/float(m_numStages) << "	"
							<< resPack->getDouble("memUsageGPU")/float(m_numStages) << "	"
							<< resPack->getDouble("cl1EnrichmentFactor")/float(m_numStages) << "	"
							<< resPack->getDouble("cl2EnrichmentFactor")/float(m_numStages) << "	"
							<< calculateStandardDeviation(resPack->getDoubleVector("totalTime")) << "\n";
				ConfigManager::writeToFile(fileMessage.str(),"CrossValidation_Output.txt");
			}

			data->m_gui->setText(IDC_STATIC_INFOTEXT,outStream.str());
		}
		outStream <<
			"================================================================================\r\n";

		data->m_gui->setText(IDC_STATIC_INFOTEXT,outStream.str());
		data->m_minerGUI->enableAllButStop();
	}

	bool CrossValidation::advance(){
		if(m_stage >= m_numStages){
			if(m_stage == m_numStages && m_useValidationSet){
			}
			else
				return false;
		}

		m_testingInds.clear();
		m_trainingInds.clear();

		std::vector<int> cl(m_clInstances.size(),0);
		if(m_numStages == m_stage && m_useValidationSet){
			for(unsigned int i=0; i<m_validationSet.size(); i++){
				m_testingInds.push_back(m_validationSet[i]->getIndex());
			}
			for(unsigned int i=0; i<m_clInstances.size(); ++i){
				for(unsigned int j=0; j<m_clInstances[i].size(); j++){
					m_trainingInds.push_back(m_clInstances[i][j]->getIndex());
				}
			}

			calculateCost(m_clInstances[0].size(),m_clInstances[1].size());
		}
		else{
			for(unsigned int i=0; i<m_clInstances.size(); i++){
				for(unsigned int j=0; j<m_clInstances[i].size(); j++){
					if(m_foldsCl[i][m_stage].find(m_clInstances[i][j]->getIndex()) != m_foldsCl[i][m_stage].end()){
						m_testingInds.push_back(m_clInstances[i][j]->getIndex());
						m_foldsCl[i][m_stage].erase(m_clInstances[i][j]->getIndex());
					}
					else{
						m_trainingInds.push_back(m_clInstances[i][j]->getIndex());
						++cl[i];
					}
				}
			}
		}

		//scrambleInstances();
		++m_stage;
		return true;
	}

	bool CrossValidation::isFinalStage(){
		if(m_numStages-1 == m_stage)
			return true;
		return false;
	}

	void CrossValidation::init(){
		for(unsigned int i=0; i<m_data->getNumInstances(); i++){
			m_clInstances[m_data->getInstance(i)->classValue()].push_back(m_data->getInstance(i));
		}
		
		unsigned int rngNr;
		boost::random::mt19937 rng;
		rng.seed(6342);

		if(m_useValidationSet){
			// Validation set 30% of the data
			for(unsigned int i=0; i<m_clInstances.size(); ++i){
				int num = int(float(m_clInstances[i].size())*0.3);
				for(unsigned int j=0; j<num; j++){
					boost::random::uniform_int_distribution<> indRand(0,m_clInstances[i].size()-1-i);
					rngNr = indRand(rng);

					m_validationSet.push_back(m_clInstances[i][rngNr]);
					m_clInstances[i][rngNr] = m_clInstances[i][m_clInstances[i].size()-1-i];
					m_clInstances[i].pop_back();
				}
			}
		}

		// Select folds
		std::vector<std::vector<unsigned int>> clSelectionVec(m_clInstances.size(),std::vector<unsigned int>());
		for(unsigned int i=0; i<m_clInstances.size(); ++i){
			clSelectionVec[i].reserve(m_clInstances[i].size());
			for(unsigned int j=0; j<m_clInstances[i].size(); ++j){
				clSelectionVec[i].push_back(j);
			}
		}

		m_foldsCl.assign(m_clInstances.size(),std::vector<std::set<unsigned int>>());
		std::vector<int> clFoldSize(m_clInstances.size(),0);
		std::vector<int> clCount(m_clInstances.size(),0);
		for(unsigned int i=0; i<m_clInstances.size(); ++i){
			clFoldSize[i] = float(m_clInstances[i].size())/float(m_numStages);
			m_foldsCl[i].assign(m_numStages,std::set<unsigned int>());
		}

		for(unsigned int i=0; i<m_numStages; ++i){
			for(unsigned int k=0; k<m_clInstances.size(); ++k){
				for(unsigned int j=0; j<clFoldSize[k]; ++j){
					boost::random::uniform_int_distribution<> indRand(0,m_clInstances[k].size()-1-clCount[k]);
					rngNr = indRand(rng);

					m_foldsCl[k][i].insert(m_clInstances[k][clSelectionVec[k][rngNr]]->getIndex());
					clSelectionVec[k][rngNr] = clSelectionVec[k][m_clInstances[k].size()-1-clCount[k]];
					++clCount[k];
				}
			}
		}

		for(unsigned int j=0; j<m_clInstances.size(); ++j){
			for(unsigned int i=0; i<m_clInstances[j].size()-clCount[j]; ++i){
				m_foldsCl[j].back().insert(m_clInstances[j][clSelectionVec[j][i]]->getIndex());
			}
		}
	}
}