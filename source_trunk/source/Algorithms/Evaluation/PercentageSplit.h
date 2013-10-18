#pragma once
#include "IEvaluation.h"

namespace DataMiner{
	class PercentageSplit : public IEvaluation{
	public:
		PercentageSplit(float splitPercentage);

		void run(IAlgorithmPtr algo, AlgorithmDataPackPtr data);

		bool advance();
		void init();
		bool isFinalStage();
	private:
		bool m_advance;
		float m_splitPercentage;
	};
}