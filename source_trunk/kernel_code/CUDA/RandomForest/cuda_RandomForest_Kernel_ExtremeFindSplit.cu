#include "cuda_RandomForest_Constants.cu"
#include "..\cuda_Common_Include.cu"
#include "..\cuda_Common_RNG.cu"

namespace ExtremeFindSplit{
	__constant__ SharedBuffer cb_constants;

	// Host function for updates to constants
	__host__ void cuda_RandomForest_UpdateConstants(void* src){
		cudaMemcpyToSymbolAsync(cb_constants,src,sizeof(SharedBuffer));
	}

	__device__ c_precision entropyConditionedOnRows(unsigned int *matrix);
	__device__ c_precision entropyOverColumns(unsigned int *matrix);
	__device__ void evaluateNumericAttribute(paramPack_Kernel* params);
	__device__ void evaluateNominalAttribute(paramPack_Kernel* params);

	__shared__ unsigned int s_attribute, s_nodeIndStart, s_numInds, s_treeId, s_attType;
	__shared__ unsigned int s_currDist[max_nominal*2];
	__shared__ c_precision s_split;

	__global__ void kernel_entry(paramPack_Kernel params){
		unsigned int nodeId = params.innerNodeIds[blockIdx.x + cb_constants.cb_currentDepth + (cb_constants.cb_nodeBufferEnd+1)*cb_constants.cb_nodeIdFlip];

		// Block initialization
		if(threadIdx.x == 0){
			if(cb_constants.cb_numFeatures == 0)
				params.splitVals[nodeId] = -FLT_MAX;

			s_treeId = params.treeIds[nodeId];
			s_split = 0;

			s_nodeIndStart = params.nodeIndicesLimits[nodeId];
			s_numInds = params.nodeIndicesLimits[(cb_constants.cb_nodeBufferEnd+1)+nodeId] - s_nodeIndStart + 1;

			stateRNG_xorShift128 state;
			state.x = params.rngStates[blockIdx.x];
			state.y = params.rngStates[blockIdx.x + (cb_constants.cb_nodeBufferEnd+1)];
			state.z = params.rngStates[blockIdx.x + (cb_constants.cb_nodeBufferEnd+1)*2];
			state.w = params.rngStates[blockIdx.x + (cb_constants.cb_nodeBufferEnd+1)*3];

			s_attribute = xorShift128(&state) % cb_constants.cb_attributeCount;
			s_attType = params.attributeTypes[s_attribute];

			if(s_attType >= max_nominal){
				unsigned int instanceInd;
				c_precision splitPoint = 0;
				for(unsigned int i=0; i<10; ++i){
					instanceInd = params.nodeIndices[s_nodeIndStart + (xorShift128(&state) % s_numInds)];
					splitPoint += params.dataset[cb_constants.cb_instanceCount*s_attribute + instanceInd];
				}
				s_split = splitPoint/10;

				s_currDist[0] = s_currDist[1] = 0;
				s_currDist[2] = s_currDist[3] = 0;
			}
		
			params.rngStates[blockIdx.x] = state.x;
			params.rngStates[blockIdx.x + (cb_constants.cb_nodeBufferEnd+1)] = state.y;
			params.rngStates[blockIdx.x + (cb_constants.cb_nodeBufferEnd+1)*2] = state.z;
			params.rngStates[blockIdx.x + (cb_constants.cb_nodeBufferEnd+1)*3] = state.w;
		}

		__syncthreads();

		if(s_attType >= max_nominal)
			evaluateNumericAttribute(&params);
		else
			evaluateNominalAttribute(&params);

		__syncthreads();

		// Block global memory writes
		if(threadIdx.x == 0){
			c_precision prior = entropyOverColumns(s_currDist);
			c_precision posterior = entropyConditionedOnRows(s_currDist);

			if(params.splitVals[nodeId] < prior-posterior){
				// Save splitpoint, attribute and distribution
				params.splitVals[nodeId] = prior-posterior;
				params.splitPoints[nodeId] = s_split;
				params.attributes[nodeId] = s_attribute;
			}
		}
	}

	__device__ void evaluateNumericAttribute(paramPack_Kernel* params){
		unsigned int window = 0;
		unsigned int numInds = s_numInds;
		unsigned int nodeIndStart = s_nodeIndStart;
		unsigned int attribute = s_attribute;
		unsigned int treeId = s_treeId;
		unsigned int weight;
		unsigned int inst;
		c_precision val;

		while(threadIdx.x + window < numInds){
			inst = params->nodeIndices[nodeIndStart + threadIdx.x + window];
			val = params->dataset[cb_constants.cb_instanceCount * attribute + inst];
			weight = params->bagWeights[treeId*cb_constants.cb_instanceCount + inst];
			
			if(val != -FLT_MAX)
				atomicAdd(&s_currDist[2*((val < s_split) ? 0 : 1)+params->classValues[inst]],weight);
			else
				atomicAdd(&s_currDist[params->classValues[inst]],weight);
			window += thread_group_size;
		}
	}

	__device__ void evaluateNominalAttribute(paramPack_Kernel* params){
		unsigned int window = 0;
		unsigned int numInds = s_numInds;
		unsigned int nodeIndStart = s_nodeIndStart;
		unsigned int attribute = s_attribute;
		unsigned int treeId = s_treeId;
		unsigned int weight;
		unsigned int inst;
		c_precision val;

		if(threadIdx.x < 40){
			s_currDist[threadIdx.x] = 0.0;
		}

		__syncthreads();

		// Split on median value
		while(threadIdx.x + window < numInds){
			inst = params->nodeIndices[nodeIndStart + threadIdx.x + window];
			val = params->dataset[cb_constants.cb_instanceCount * attribute + inst];
			weight = params->bagWeights[treeId*cb_constants.cb_instanceCount + inst];

			if(val != -FLT_MAX)
				atomicAdd(&s_currDist[2*int(val)+params->classValues[inst]],weight);
			else
				atomicAdd(&s_currDist[params->classValues[inst]],weight);
			window += thread_group_size;
		}
	}

	__device__ c_precision lnFunc(c_precision num){
		if(num <= 1e-6){
			return 0;
		} 
		else{
			return num * log(num);
		}
	}

	__device__ c_precision entropyConditionedOnRows(unsigned int *matrix){
		unsigned int nodes = (s_attType >= max_nominal) ? 2 : s_attType;

		c_precision returnValue = 0, sumForRow, total = 0;
		for (int i = 0; i < nodes; i++) {
			sumForRow = 0;
			for (int j = 0; j < 2; j++) {
				returnValue = returnValue + lnFunc(matrix[2*i+j]);
				sumForRow += matrix[2*i+j];
			}
			returnValue = returnValue - lnFunc(sumForRow);
			total += sumForRow;
		}
		if(total < 1.0e-6) {
			return 0;
		}
		return -returnValue / (total * log(c_precision(2.0)));
	}

	__device__ c_precision entropyOverColumns(unsigned int *matrix){
		unsigned int nodes = (s_attType >= max_nominal) ? 2 : s_attType;
		
		c_precision returnValue = 0, sumForColumn, total = 0;
		for (int j = 0; j < 2; j++){
			sumForColumn = 0;
			for(int i = 0; i < nodes; i++){
				sumForColumn += matrix[2*i+j];
			}
			returnValue = returnValue - lnFunc(sumForColumn);
			total += sumForColumn;
		}
		if(total < 1.0e-6){
		  return 0;
		}
		return (returnValue + lnFunc(total)) / (total * log(c_precision(2.0)));
	}
}