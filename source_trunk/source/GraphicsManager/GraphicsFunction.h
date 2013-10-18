#pragma once
#include "GraphicsManager.h"
#include "CUDAManager.h"

namespace DataMiner{
	class GraphicsFunction{
	public:
		virtual void run(int devId, std::vector<int> params, GraphicsManagerPtr gfxMgr, int x, int y, int z, void* data = NULL) = 0;
	private:
	};
}