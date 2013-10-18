#pragma once
#include "GPURandomForest.h"

namespace DataMiner{
	class GPUERT : public GPURandomForest{
	public:
		GPUERT(GraphicsManagerPtr gfxMgr);
		~GPUERT();

		void runBuildProcess(int devId, int trees, BarrierPtr bar);
	private:
		void deviceHandler();

		ThreadPtr m_thread;
		BarrierPtr m_bar;
	};
}

typedef boost::shared_ptr<DataMiner::GPUERT> GPUERTPtr;