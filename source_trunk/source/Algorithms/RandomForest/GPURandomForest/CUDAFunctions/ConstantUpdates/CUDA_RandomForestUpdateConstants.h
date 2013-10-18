#pragma once
#include "GraphicsFunction.h"
#include "GPURandomForest.h"

namespace DataMiner{
	extern "C" void cuda_RandomForest_Wrapper_UpdateConstants(void* src, int id);

	class CUDA_RandomForestUpdateConstants : public GraphicsFunction{
	public:
		void run(int devId, std::vector<int> params, GraphicsManagerPtr gfxMgr, int x, int y, int z, void* data = NULL){
			CUDAManagerPtr cMgr = boost::static_pointer_cast<CUDAManager>(gfxMgr);

			cuda_RandomForest_Wrapper_UpdateConstants(((GPURandomForest::ConstantUpdate*)data)->m_content,((GPURandomForest::ConstantUpdate*)data)->m_id);
		}
	private:
	};
}