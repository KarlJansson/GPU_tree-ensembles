#pragma once
#include "FrameworkMessage.h"

namespace DataMiner{
	class StartAlgorithmMessage : public FrameworkMessage{
	public:
		StartAlgorithmMessage(IDataPackPtr datapack, int barrier = 0):FrameworkMessage("StartAlgorithm",datapack,barrier){}
	private:
	};
}

typedef boost::shared_ptr<DataMiner::StartAlgorithmMessage> StartAlgorithmMessagePtr;