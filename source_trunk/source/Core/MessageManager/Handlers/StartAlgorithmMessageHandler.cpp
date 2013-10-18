#include "stdafx.h"
#include "StartAlgorithmMessageHandler.h"
#include "RandomForest.h"
#include "CrossValidation.h"
#include "PercentageSplit.h"
#include "NoEvaluation.h"
#include "GPURandomForest.h"
#include "ExRandomTrees.h"

namespace DataMiner{
	StartAlgorithmMessageHandler::StartAlgorithmMessageHandler(GraphicsManagerPtr gfxMgr){
	}

	void StartAlgorithmMessageHandler::stop(){
		m_runningAlgo->stop();
	}

	void StartAlgorithmMessageHandler::handle(IDataPackPtr dataPack){
		AlgorithmDataPackPtr data = boost::static_pointer_cast<AlgorithmDataPack>(dataPack);
		selectEvaluation(data);
		selectAlgorithm(data);
		m_evaluation->run(m_runningAlgo,data);

		m_evaluation.reset();
		m_runningAlgo.reset();
	}

	void StartAlgorithmMessageHandler::selectEvaluation(AlgorithmDataPackPtr data){
		std::string eval = data->m_gui->getEditText(IDC_COMBO_EVALUATION);
		if(eval.compare("CrossValidation") == 0){
			unsigned int folds = 0;
			try{
				folds = boost::lexical_cast<unsigned int>(data->m_gui->getEditText(IDC_EDIT_EVALPARAM));
			}catch(...){
				folds = 10;
			}
			m_evaluation = IEvaluationPtr(new CrossValidation(folds));
		}
		else if(eval.compare("PercentageSplit") == 0){
			float percent = 0;
			try{
				percent = boost::lexical_cast<float>(data->m_gui->getEditText(IDC_EDIT_EVALPARAM));
			}catch(...){
				percent = 66;
			}
			m_evaluation = IEvaluationPtr(new PercentageSplit(percent));
		}
		else{
			m_evaluation = IEvaluationPtr(new NoEvaluation);
		}
	}

	void StartAlgorithmMessageHandler::selectAlgorithm(AlgorithmDataPackPtr data){
		if(data->m_algoName.compare("CPURandomForest") == 0){
			m_runningAlgo = IAlgorithmPtr(new ExRandomTrees());
		}
		else if(data->m_algoName.compare("GPURandomForest") == 0){
			m_runningAlgo = IAlgorithmPtr(new GPURandomForest(data->m_gfxMgr));
		}
	}
}