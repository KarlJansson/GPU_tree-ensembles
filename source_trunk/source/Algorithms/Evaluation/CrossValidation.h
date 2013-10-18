#pragma once
#include "IEvaluation.h"

namespace DataMiner{
	class CrossValidation : public IEvaluation{
	public:
		CrossValidation(unsigned int folds);

		void run(IAlgorithmPtr algo, AlgorithmDataPackPtr data);

		bool advance();
		void init();
		bool isFinalStage();
	private:
		bool m_useValidationSet;
		std::vector<InstancePtr> m_validationSet;
		std::vector<std::vector<std::set<unsigned int>>> m_foldsCl;
	};
}

typedef boost::shared_ptr<DataMiner::CrossValidation> CrossValidationPtr;