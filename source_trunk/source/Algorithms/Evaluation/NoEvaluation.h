#pragma once
#include "IEvaluation.h"

namespace DataMiner{
	class NoEvaluation : public IEvaluation{
	public:
		void run(IAlgorithmPtr algo, AlgorithmDataPackPtr data);

		bool advance();
		void init();
		bool isFinalStage();
	private:
	};
}