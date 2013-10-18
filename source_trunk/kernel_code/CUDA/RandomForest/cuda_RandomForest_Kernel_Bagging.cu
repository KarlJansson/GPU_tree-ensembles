#include "cuda_RandomForest_Constants.cu"
#include "..\cuda_Common_Include.cu"
#include "..\cuda_Common_RNG.cu"

namespace Bagging{
	__constant__ ConstantBufferBagging cb_constants;

	__shared__ unsigned int s_bagCount;
	__shared__ unsigned int s_classProb[2];
	__shared__ stateRNG_xorShift128 s_state;
	__shared__ bool s_addIntent[thread_group_size];

	// Host function for updates to constants
	__host__ void cuda_RandomForest_UpdateConstants(void* src){
		cudaMemcpyToSymbol(Bagging::cb_constants,src,sizeof(ConstantBufferBagging));
	}
	__device__ bool withBagging(paramPack_Kernel &params);
	__device__ bool withoutBagging(paramPack_Kernel &params);

	__global__ void kernel_entry(paramPack_Kernel params){
		unsigned int treeId = cb_constants.cb_treeOffset + blockIdx.x;
		bool success;

		success = cb_constants.cb_baggingActivated ? withBagging(params) : withoutBagging(params);

		// Write result to memory
		if(threadIdx.x == 0 && success){
			params.classProbs[treeId] = s_classProb[0];
			params.classProbs[(cb_constants.cb_nodeBufferEnd+1)+treeId] = s_classProb[1];
			params.treeIds[treeId] = treeId;
			params.nodeIndicesLimits[treeId] = treeId*cb_constants.cb_instanceCount;
			params.nodeIndicesLimits[(cb_constants.cb_nodeBufferEnd+1)+treeId] = treeId*cb_constants.cb_instanceCount+s_bagCount-1;
		}
	}

	__device__ bool withBagging(paramPack_Kernel &params){
		unsigned int numValForCore = cb_constants.cb_instanceCount < thread_group_size ? 0 : cb_constants.cb_instanceCount / thread_group_size;
		if(threadIdx.x < cb_constants.cb_instanceCount % thread_group_size)
			++numValForCore;
		if(numValForCore == 0)
			return false;

		if(threadIdx.x == 0){
			s_bagCount = 0;
			s_classProb[1] = s_classProb[0] = 0;

			s_state.x = params.stateBuffer[blockIdx.x];
			s_state.y = params.stateBuffer[blockIdx.x+gridDim.x];
			s_state.z = params.stateBuffer[blockIdx.x+gridDim.x*2];
			s_state.w = params.stateBuffer[blockIdx.x+gridDim.x*3];
		}

		__syncthreads();

		unsigned int treeId = cb_constants.cb_treeOffset + blockIdx.x;
		unsigned int tid = blockIdx.x*thread_group_size+threadIdx.x;

		stateRNG_xorShift128 state;
		state.x = s_state.x+threadIdx.x;
		state.y = s_state.y+threadIdx.x;
		state.z = s_state.z+threadIdx.x;
		state.w = s_state.w+threadIdx.x;

		unsigned int randVal;
		unsigned int weight;
		for(unsigned int i=0; i<numValForCore; ++i){
			randVal = xorShift128(&state) % cb_constants.cb_instanceCount;
			weight = atomicAdd(&params.bagWeights[treeId * cb_constants.cb_instanceCount + randVal],1);

			s_addIntent[threadIdx.x] = weight == 0 ? true : false;

			__syncthreads();

			// Add new bag indice
			if(s_addIntent[threadIdx.x]){
				unsigned int offset = 0;
				for(unsigned int j=0; j<threadIdx.x; ++j){
					offset += s_addIntent[j] ? 1 : 0;
				}

				params.nodeIndices[cb_constants.cb_instanceCount*treeId + s_bagCount + offset] = randVal;
				params.nodeIndicesMirror[cb_constants.cb_instanceCount*treeId + s_bagCount + offset] = randVal;
			}

			__syncthreads();
			if(s_addIntent[threadIdx.x]){
				atomicAdd(&s_bagCount,1);
				atomicAdd(&s_classProb[params.classValues[randVal]],1);
			}
			__syncthreads();
		}


		if(threadIdx.x == 0){
			params.stateBuffer[blockIdx.x] = state.x;
			params.stateBuffer[blockIdx.x+gridDim.x] = state.y;
			params.stateBuffer[blockIdx.x+gridDim.x*2] = state.z;
			params.stateBuffer[blockIdx.x+gridDim.x*3] = state.w;
		}

		return true;
	}

	__device__ bool withoutBagging(paramPack_Kernel &params){
		if(threadIdx.x == 0){
			s_bagCount = cb_constants.cb_instanceCount;
			s_classProb[1] = s_classProb[0] = 0;
		}

		__syncthreads();

		unsigned int treeId = cb_constants.cb_treeOffset + blockIdx.x;
		unsigned int window = 0;
		while(threadIdx.x+blockDim.x*window < cb_constants.cb_instanceCount){
			params.nodeIndices[cb_constants.cb_instanceCount*treeId + threadIdx.x+blockDim.x*window] = threadIdx.x+blockDim.x*window;
			params.nodeIndicesMirror[cb_constants.cb_instanceCount*treeId + threadIdx.x+blockDim.x*window] = threadIdx.x+blockDim.x*window;
			params.bagWeights[treeId*cb_constants.cb_instanceCount + threadIdx.x+blockDim.x*window] = 1;
			atomicAdd(&s_classProb[params.classValues[threadIdx.x+blockDim.x*window]],1);
			++window;
		}

		return true;
	}
}

