#include "cuda_RandomForest_Constants.cu"
#include "..\cuda_Common_Include.cu"
#include "..\cuda_Common_RNG.cu"

namespace RandomForest_SplitData_Kernel{
	__constant__ SharedBuffer cb_constants;

	// Host function for updates to constants
	__host__ void cuda_RandomForest_UpdateConstants(void* src){
		cudaMemcpyToSymbolAsync(cb_constants,src,sizeof(SharedBuffer));
	}

	__device__ void splitNumericAttribute(paramPack_Kernel *params);
	__device__ void splitNominalAttribute(paramPack_Kernel *params);

	__shared__ unsigned int s_numNodeIndices, s_nodeId, s_nodeIndStart, s_num[max_nominal*2], s_attType;
	__shared__ int s_bestAttribute;
	__shared__ c_precision s_bestSplit;

	__global__ void kernel_entry(paramPack_Kernel params){
		if(threadIdx.x == 0){
			s_nodeId = params.innerNodeIds[blockIdx.x + cb_constants.cb_currentDepth + (cb_constants.cb_nodeBufferEnd+1)*cb_constants.cb_nodeIdFlip];
			s_nodeIndStart = params.nodeIndicesLimits[s_nodeId];
			s_numNodeIndices = params.nodeIndicesLimits[s_nodeId+(cb_constants.cb_nodeBufferEnd+1)]-s_nodeIndStart+1;
			s_bestAttribute = params.attributeBuffer[s_nodeId];
			s_attType = params.attributeNumValues[s_bestAttribute];
			s_bestSplit = params.splitPoints[s_nodeId];
		}

		__syncthreads();
		
		if(s_attType >= max_nominal)
			splitNumericAttribute(&params);
		else
			splitNominalAttribute(&params);
	}

	__device__ void splitNumericAttribute(paramPack_Kernel *params){
		int bestAtt = s_bestAttribute;
		if(bestAtt < 0)
			bestAtt = -3;

		unsigned int pos, nodeInd;
		unsigned int window = 0;
		unsigned int numNodeIndices = s_numNodeIndices;
		unsigned int nodeIndStart = s_nodeIndStart;
		c_precision bestSplit = s_bestSplit;
		c_precision datPoint;

		if(threadIdx.x == 0){
			s_num[0] = 0;
			s_num[1] = numNodeIndices-1;
		}

		__syncthreads();

		while(threadIdx.x+window < numNodeIndices){
			nodeInd = params->nodeIndices[nodeIndStart+threadIdx.x+window];

			if(bestAtt != -3){
				datPoint = params->inputData[bestAtt*cb_constants.cb_instanceCount+params->inputInds[bestAtt*cb_constants.cb_instanceCount+nodeInd]];
				pos = (datPoint < bestSplit) ? atomicAdd(&s_num[0],1) : atomicAdd(&s_num[1],-1);
			}
			else
				pos = threadIdx.x+window;

			pos += nodeIndStart;
			params->nodeIndicesMirror[pos] = nodeInd;
			window += thread_group_size;
		}
	}

	__device__ void splitNominalAttribute(paramPack_Kernel *params){
		int bestAtt = s_bestAttribute;
		if(bestAtt < 0)
			bestAtt = -3;

		if(threadIdx.x < (s_attType >= max_nominal ? 4 : s_attType*2)){
			s_num[threadIdx.x] = params->distBuffer[s_nodeId+(cb_constants.cb_nodeBufferEnd+1)*threadIdx.x];
		}

		__syncthreads();

		if(threadIdx.x == 0){
			unsigned int num[max_nominal];
			unsigned int itr = (s_attType >= max_nominal ? 2 : s_attType);
			for(unsigned int i=0; i<itr; i++){
				num[i] = s_num[2*i]+s_num[(2*i)+1];
			}

			unsigned int add = num[0];
			unsigned int tmp;
			num[0] = 0;
			for(unsigned int i=1; i<itr; ++i){
				tmp = num[i];
				num[i] = num[i-1] + add;
				add = tmp;
			}

			for(unsigned int i=0; i<itr; i++){
				s_num[i] = num[i];
			}
		}

		__syncthreads();

		unsigned int pos, nodeInd;
		unsigned int window = 0;
		unsigned int numNodeIndices = s_numNodeIndices;
		unsigned int nodeIndStart = s_nodeIndStart;
		unsigned int datPoint;

		while(threadIdx.x+window < numNodeIndices){
			nodeInd = params->nodeIndices[nodeIndStart+threadIdx.x+window];
			
			if(bestAtt != -3){
				datPoint = params->inputData[bestAtt*cb_constants.cb_instanceCount+params->inputInds[bestAtt*cb_constants.cb_instanceCount+nodeInd]];
				pos = atomicAdd(&s_num[datPoint],1);
			}
			else
				pos = threadIdx.x+window;

			pos += nodeIndStart;
			params->nodeIndicesMirror[pos] = nodeInd;
			window += thread_group_size;
		}
	}
}