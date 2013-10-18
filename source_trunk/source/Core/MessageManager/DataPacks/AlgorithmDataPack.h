#pragma once
#include "IDataPack.h"

namespace DataMiner{
	class AlgorithmDataPack : public IDataPack{
	public:
		std::string m_algoName;
		std::string m_dataResource;
	private:
	};
}

typedef boost::shared_ptr<DataMiner::AlgorithmDataPack> AlgorithmDataPackPtr;