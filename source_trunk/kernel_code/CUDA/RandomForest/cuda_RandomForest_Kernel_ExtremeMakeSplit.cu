#include "cuda_RandomForest_Constants.cu"
#include "..\cuda_Common_Include.cu"
#include "..\cuda_Common_RNG.cu"

namespace ExtremeMakeSplit{
	__constant__ SharedBuffer cb_constants;

	// Host function for updates to constants
	__host__ void cuda_RandomForest_UpdateConstants(void* src){
		cudaMemcpyToSymbolAsync(cb_constants,src,sizeof(SharedBuffer));
	}

	__device__ void splitNumeric(paramPack_Kernel *params);
	__device__ void splitNominal(paramPack_Kernel *params);
	__device__ void assembleNumericHistogram(paramPack_Kernel *params);
	__device__ void assembleNominalHistogram(paramPack_Kernel *params);

	__shared__ unsigned int s_numNodeIndices, s_nodeIndStart, s_num[max_nominal*2], s_attType, s_distStart;
	__shared__ int s_bestAttribute;
	__shared__ c_precision s_bestSplit;

	__global__ void kernel_entry(paramPack_Kernel params){
		unsigned int nodeId = params.innerNodeIds[blockIdx.x + cb_constants.cb_currentDepth + (cb_constants.cb_nodeBufferEnd+1)*cb_constants.cb_nodeIdFlip];
		unsigned int itr;

		if(threadIdx.x == 0){
			s_nodeIndStart = params.nodeIndicesLimits[nodeId];
			unsigned int nodeIndEnd = params.nodeIndicesLimits[nodeId+(cb_constants.cb_nodeBufferEnd+1)];
			s_numNodeIndices = nodeIndEnd-s_nodeIndStart+1;
			s_bestAttribute = params.attributes[nodeId];
			s_attType = params.attributeTypes[s_bestAttribute];
			s_bestSplit = params.splitPoints[nodeId];
			s_distStart = atomicAdd(&params.check[1],(s_attType >= max_nominal ? 4 : s_attType*2));
			params.childIds[nodeId] = s_distStart;
		}

		__syncthreads();

		if(threadIdx.x < (s_attType >= max_nominal ? 4 : s_attType*2)){
			s_num[threadIdx.x] = 0;
		}

		__syncthreads();
		
		// Count instances per node
		if(s_attType >= max_nominal)
			assembleNumericHistogram(&params);
		else
			assembleNominalHistogram(&params);

		__syncthreads();

		// Save node counts
		itr = (s_attType >= max_nominal ? 4 : s_attType*2);
		for(unsigned int i=0; i<itr; ++i){
			params.distBuffer[s_distStart+i] = s_num[i];
		}

		__syncthreads();

		// Asssemble histogram for node split
		if(threadIdx.x == 0){
			unsigned int num[max_nominal];
			unsigned int itr = (s_attType >= max_nominal ? 2 : s_attType);

			// Hack to force the loop to not get optimized away. :S
			// ------------------------------
			num[0] = s_num[0] + s_num[1];
			num[1] = s_num[2] + s_num[3];
			// ------------------------------
			for(unsigned int i=0; i<itr; i++){
				num[i] = s_num[2*i]+s_num[(2*i)+1];
			}

			unsigned int add = num[0];
			unsigned int tmp;
			num[0] = 0;
			for(unsigned int i=1; i<itr; i++){
				tmp = num[i];
				num[i] = num[i-1] + add;
				add = tmp;
			}

			for(unsigned int i=0; i<itr; i++){
				s_num[i] = num[i];
			}
		}

		__syncthreads();
		
		// Perform split
		if(s_attType >= max_nominal)
			splitNumeric(&params);
		else
			splitNominal(&params);
	}

	__device__ void assembleNumericHistogram(paramPack_Kernel *params){
		unsigned int window = 0;
		unsigned int numInds = s_numNodeIndices;
		unsigned int nodeIndStart = s_nodeIndStart;
		unsigned int attribute = s_bestAttribute;
		unsigned int inst;
		c_precision val;

		while(threadIdx.x + window < numInds){
			inst = params->nodeIndices[nodeIndStart + threadIdx.x + window];
			val = params->dataset[cb_constants.cb_instanceCount * attribute + inst];
			
			if(val != -FLT_MAX)
				atomicAdd(&s_num[2*((val < s_bestSplit) ? 0 : 1)+params->classValues[inst]],1);
			else
				atomicAdd(&s_num[params->classValues[inst]],1);
			window += thread_group_size;
		}
	}

	__device__ void assembleNominalHistogram(paramPack_Kernel *params){
		unsigned int window = 0;
		unsigned int numInds = s_numNodeIndices;
		unsigned int nodeIndStart = s_nodeIndStart;
		unsigned int attribute = s_bestAttribute;
		unsigned int inst;
		c_precision val;

		while(threadIdx.x + window < numInds){
			inst = params->nodeIndices[nodeIndStart + threadIdx.x + window];
			val = params->dataset[cb_constants.cb_instanceCount * attribute + inst];

			if(val != -FLT_MAX)
				atomicAdd(&s_num[2*int(val)+params->classValues[inst]],1);
			else
				atomicAdd(&s_num[params->classValues[inst]],1);
			window += thread_group_size;
		}
	}

	__device__ void splitNumeric(paramPack_Kernel *params){
		int bestAtt = s_bestAttribute;
		if(bestAtt < 0)
			bestAtt = -3;

		unsigned int pos, nodeInd;
		unsigned int window = 0;
		unsigned int numNodeIndices = s_numNodeIndices;
		unsigned int nodeIndStart = s_nodeIndStart;
		c_precision bestSplit = s_bestSplit;
		c_precision datPoint;

		while(threadIdx.x+window < numNodeIndices){
			nodeInd = params->nodeIndices[nodeIndStart+threadIdx.x+window];

			if(bestAtt != -3){
				datPoint = params->dataset[bestAtt*cb_constants.cb_instanceCount+nodeInd];
				if(datPoint != -FLT_MAX)
					pos = (datPoint < bestSplit) ? atomicAdd(&s_num[0],1) : atomicAdd(&s_num[1],1);
				else
					pos = atomicAdd(&s_num[0],1);
			}
			else
				pos = threadIdx.x+window;

			pos += nodeIndStart;
			params->nodeIndicesMirror[pos] = nodeInd;
			window += thread_group_size;
		}
	}

	__device__ void splitNominal(paramPack_Kernel *params){
		int bestAtt = s_bestAttribute;
		if(bestAtt < 0)
			bestAtt = -3;

		unsigned int pos, nodeInd;
		unsigned int window = 0;
		unsigned int numNodeIndices = s_numNodeIndices;
		unsigned int nodeIndStart = s_nodeIndStart;
		c_precision bestSplit = s_bestSplit;
		c_precision datPoint;

		while(threadIdx.x+window < numNodeIndices){
			nodeInd = params->nodeIndices[nodeIndStart+threadIdx.x+window];
			
			if(bestAtt != -3){
				datPoint = params->dataset[bestAtt*cb_constants.cb_instanceCount+nodeInd];
				if(datPoint != -FLT_MAX)
					pos = atomicAdd(&s_num[int(datPoint)],1);
				else
					pos = atomicAdd(&s_num[0],1);
			}
			else
				pos = threadIdx.x+window;

			pos += nodeIndStart;
			params->nodeIndicesMirror[pos] = nodeInd;
			window += thread_group_size;
		}
	}
}