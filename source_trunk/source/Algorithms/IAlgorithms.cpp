#include "stdafx.h"
#include "IAlgorithm.h"
#include "CrossValidation.h"
#include "PercentageSplit.h"
#include "NoEvaluation.h"

namespace DataMiner{
	void IAlgorithm::runAlgorithm(AlgorithmDataPackPtr data, IEvaluationPtr eval){
		m_data = data;
		m_document = m_data->m_recMgr->getDocumentResource(m_data->m_dataResource);

		if(m_document){
			m_evaluation = eval;
			run();
		}
		else{
			std::wstringstream wss;
			wss << "Specified data document not found.\r\n";
			m_data->m_gui->postDebugMessage(wss.str());
		}

		if(m_data->m_callBack){
			(*m_data->m_callBack)(shared_from_this());
		}
	}
}