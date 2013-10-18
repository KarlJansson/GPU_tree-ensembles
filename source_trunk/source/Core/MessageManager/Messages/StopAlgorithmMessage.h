#pragma once
#include "FrameworkMessage.h"

namespace DataMiner{
	class StopAlgorithmMessage : public FrameworkMessage{
	public:
		StopAlgorithmMessage(IDataPackPtr datapack, int barrier = 0):FrameworkMessage("StopAlgorithm",datapack,barrier){}
	private:
	};
}

typedef boost::shared_ptr<DataMiner::StopAlgorithmMessage> StopAlgorithmMessagePtr;