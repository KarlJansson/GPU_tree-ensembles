#pragma once
#include "GraphicsFunction.h"

namespace DataMiner{
	extern "C" void cuda_RandomForest_Wrapper_FindSplit(unsigned int num_threads, void* params);

	class CUDA_RandomForestFindSplit : public GraphicsFunction{
	private:
		struct paramPackage{
			void* params[20];
		};
	public:
		void run(int devId, std::vector<int> params, GraphicsManagerPtr gfxMgr, int x, int y, int z, void* data = NULL){
			CUDAManagerPtr cMgr = boost::static_pointer_cast<CUDAManager>(gfxMgr);

			paramPackage* paramPack = new paramPackage;
			for(unsigned int i=0; i<params.size(); ++i){
				paramPack->params[i] = cMgr->getResource(devId, params[i]);
			}

			cuda_RandomForest_Wrapper_FindSplit(x*y*z,paramPack);
			delete paramPack;
		}
	};
}