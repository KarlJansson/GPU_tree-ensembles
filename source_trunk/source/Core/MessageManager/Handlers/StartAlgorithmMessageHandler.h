#pragma once
#include "MessageHandler.h"
#include "AlgorithmDataPack.h"
#include "IAlgorithm.h"

namespace DataMiner{
	class StartAlgorithmMessageHandler : public MessageHandler{
	public:
		StartAlgorithmMessageHandler(GraphicsManagerPtr gfxMgr);

		void handle(IDataPackPtr dataPack);
		void stop();
	private:
		void selectEvaluation(AlgorithmDataPackPtr data);
		void selectAlgorithm(AlgorithmDataPackPtr data);

		std::map<std::string,IAlgorithmPtr> m_algorithms;
		IAlgorithmPtr m_runningAlgo;
		IEvaluationPtr m_evaluation;
	};
}