#pragma once
#include "GPURandomForest.h"

namespace DataMiner{
	class GPURF : public GPURandomForest{
	public:
		GPURF(GraphicsManagerPtr gfxMgr);
		~GPURF();

		void runBuildProcess(int devId, int trees, BarrierPtr bar);
	private:
		void deviceHandler();

		ThreadPtr m_thread;
		BarrierPtr m_bar;
	};
}

typedef boost::shared_ptr<DataMiner::GPURF> GPURFPtr;