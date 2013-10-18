#pragma once
#include "Runnable.h"
#include "MainFramework.h"
#include "StopAlgorithmMessage.h"

namespace DataMiner{
	class ButtonStopAlgorithm : public Runnable{
	public:
		void run(void* params){
			MainFramework *mainFramework = (MainFramework*)params;
			AlgorithmDataPackPtr data = AlgorithmDataPackPtr(new AlgorithmDataPack);
			mainFramework->postMessage(StopAlgorithmMessagePtr(new StopAlgorithmMessage(data)));
		}
	private:
	};
}