#include "cuda_RandomForest_Constants.cu"
#include "..\cuda_Common_Include.cu"
#include "..\cuda_Common_RNG.cu"

namespace FindSplit{
	__constant__ SharedBuffer cb_constants;

	// Host function for updates to constants
	__host__ void cuda_RandomForest_UpdateConstants(void* src){
		cudaMemcpyToSymbolAsync(cb_constants,src,sizeof(SharedBuffer));
	}

	// Forward declaration
	__device__ c_precision entropyConditionedOnRows(unsigned int *matrix);
	__device__ c_precision entropyOverColumns(unsigned int *matrix);

	__device__ void evaluateNumericAttribute(paramPack_Kernel *params);
	__device__ void evaluateNominalAttribute(paramPack_Kernel *params);

	struct sharedDataParams{
		unsigned int nodeId;
		unsigned int treeId;
		unsigned int nodeIndStart;
		unsigned int numInds;
		unsigned int attribute;

		unsigned int accWeight1;
		unsigned int accWeight2;
	};

	__shared__ sharedDataParams s_dataParams;

	__shared__ unsigned int s_currDist[40];
	__shared__ unsigned int s_indBuffer[thread_group_size];
	__shared__ unsigned int s_weightBuffer[thread_group_size*2];
	__shared__ c_precision s_bestVal[thread_group_size];
	__shared__ int s_bestI[thread_group_size];
	__shared__ unsigned int s_numValues;

	__global__ void kernel_entry(paramPack_Kernel params){
		if(threadIdx.x == 0){
			s_dataParams.nodeId = params.innerNodeIds[cb_constants.cb_currentDepth + blockIdx.x + (cb_constants.cb_nodeBufferEnd+1)*cb_constants.cb_nodeIdFlip];
			s_dataParams.treeId = params.treeIds[s_dataParams.nodeId];
			s_dataParams.nodeIndStart = params.nodeIndicesLimits[s_dataParams.nodeId];
			s_dataParams.numInds = params.nodeIndicesLimits[(cb_constants.cb_nodeBufferEnd+1)+s_dataParams.nodeId] - s_dataParams.nodeIndStart + 1;
			s_dataParams.accWeight1 = 0;
			s_dataParams.accWeight2 = 0;

			if(cb_constants.cb_numFeatures == 0)
				params.splitVal[s_dataParams.nodeId] = -FLT_MAX;

			stateRNG_xorShift128 state;
			state.w = params.stateBuffer[blockIdx.x];
			state.x = params.stateBuffer[(cb_constants.cb_nodeBufferEnd+1)+blockIdx.x];
			state.y = params.stateBuffer[(cb_constants.cb_nodeBufferEnd+1)*2+blockIdx.x];
			state.z = params.stateBuffer[(cb_constants.cb_nodeBufferEnd+1)*3+blockIdx.x];
			
			s_dataParams.attribute = xorShift128(&state) % cb_constants.cb_attributeCount;
			s_numValues = params.attributeNumValues[s_dataParams.attribute];

			params.stateBuffer[blockIdx.x] = state.w;
			params.stateBuffer[(cb_constants.cb_nodeBufferEnd+1)+blockIdx.x] = state.x;
			params.stateBuffer[(cb_constants.cb_nodeBufferEnd+1)*2+blockIdx.x] = state.y;
			params.stateBuffer[(cb_constants.cb_nodeBufferEnd+1)*3+blockIdx.x] = state.z;

			s_bestI[0] = 1;

			s_currDist[0] = s_currDist[1] = 0;
			s_currDist[2] = s_currDist[3] = 0;
		}
		s_bestVal[threadIdx.x] = -FLT_MAX;

		__syncthreads();

		if(s_numValues >= max_nominal)
			evaluateNumericAttribute(&params);
		else
			evaluateNominalAttribute(&params);

		__syncthreads();

		if(threadIdx.x == 0){
			c_precision prior = entropyOverColumns(s_currDist);
			c_precision posterior = entropyConditionedOnRows(s_currDist);

			if(params.splitVal[s_dataParams.nodeId] < prior-posterior){
				c_precision splitPoint = 0;
				if(s_numValues >= max_nominal && s_bestI[0] > 0){
					int instJustBeforeSplit = params.nodeIndices[s_dataParams.nodeIndStart+s_bestI[0]-1];
					int instJustAfterSplit = params.nodeIndices[s_dataParams.nodeIndStart+s_bestI[0]];
					splitPoint = ( params.inputData[s_dataParams.attribute*cb_constants.cb_instanceCount + params.inputInds[s_dataParams.attribute*cb_constants.cb_instanceCount+instJustAfterSplit]]
						+ params.inputData[s_dataParams.attribute*cb_constants.cb_instanceCount + params.inputInds[s_dataParams.attribute*cb_constants.cb_instanceCount+instJustBeforeSplit]] ) / 2.0;
				}

				// Save splitpoint, attribute and distribution
				params.splitVal[s_dataParams.nodeId] = prior-posterior;
				params.splitPoints[s_dataParams.nodeId] = splitPoint;
				params.attributeBuffer[s_dataParams.nodeId] = s_dataParams.attribute;

				for(unsigned int i=0; i<(s_numValues >= max_nominal ? 4 : s_numValues*2); ++i){
					params.distBuffer[s_dataParams.nodeId+(cb_constants.cb_nodeBufferEnd+1)*i] = s_currDist[i];
				}
			}
		}
	}

	__device__ void evaluateNumericAttribute(paramPack_Kernel *params){
		unsigned int currDist[4];

		// Find best split
		unsigned int window = 0;
		int classVal;
		c_precision currVal = -FLT_MAX;
		bool lastLane = false;
		bool syncThreads = false;
		unsigned int inst;

		while(threadIdx.x+window < s_dataParams.numInds){
			inst = params->nodeIndices[s_dataParams.nodeIndStart + window + threadIdx.x];
			atomicAdd(&s_currDist[2+params->classValues[inst]],1);
			window += thread_group_size;
		}

		__syncthreads();
		currDist[0] = currDist[1] = 0;
		currDist[2] = s_currDist[2];
		currDist[3] = s_currDist[3];

		window = 0;
		while(threadIdx.x+window < s_dataParams.numInds-1){
			syncThreads = int(s_dataParams.numInds-1) - int(window) > 32 ? true : false;
			lastLane = threadIdx.x+window == s_dataParams.numInds-2 ? true : (threadIdx.x == blockDim.x-1 ? true : false);

			s_indBuffer[threadIdx.x] = params->nodeIndices[s_dataParams.nodeIndStart + window + threadIdx.x];
			classVal = params->classValues[s_indBuffer[threadIdx.x]];
		
			if(syncThreads)
				__syncthreads();
		
			inst = lastLane ? params->nodeIndices[s_dataParams.nodeIndStart + window + threadIdx.x + 1] : s_indBuffer[threadIdx.x+1];
			s_weightBuffer[threadIdx.x*2] = 0.0f;
			s_weightBuffer[threadIdx.x*2+1] = 0.0f;
			s_weightBuffer[threadIdx.x*2+classVal] = 1;//bagWeights[s_dataParams.treeId*cb_constants.cb_instanceCount + s_indBuffer[threadIdx.x]];
		
			if(syncThreads)
				__syncthreads();

			// Accumulate weights
			if(lastLane){
				for(unsigned int i=1; i<threadIdx.x+1; i++){
					s_weightBuffer[i*2] += s_weightBuffer[i*2-2];
					s_weightBuffer[i*2+1] += s_weightBuffer[i*2-1];
				
					s_weightBuffer[i*2-2] += s_dataParams.accWeight1;
					s_weightBuffer[i*2-1] += s_dataParams.accWeight2;
				}
				s_weightBuffer[(threadIdx.x+1)*2-2] += s_dataParams.accWeight1;
				s_weightBuffer[(threadIdx.x+1)*2-1] += s_dataParams.accWeight2;
				s_dataParams.accWeight1 = s_weightBuffer[(threadIdx.x+1)*2-2];
				s_dataParams.accWeight2 = s_weightBuffer[(threadIdx.x+1)*2-1];
			}

			if(syncThreads)
				__syncthreads();

			currDist[0] += s_weightBuffer[threadIdx.x*2];
			currDist[2] -= s_weightBuffer[threadIdx.x*2];
			currDist[1] += s_weightBuffer[threadIdx.x*2+1];
			currDist[3] -= s_weightBuffer[threadIdx.x*2+1];

			if(	params->inputData[s_dataParams.attribute*cb_constants.cb_instanceCount + params->inputInds[s_dataParams.attribute*cb_constants.cb_instanceCount+inst]] > 
				params->inputData[s_dataParams.attribute*cb_constants.cb_instanceCount + params->inputInds[s_dataParams.attribute*cb_constants.cb_instanceCount+s_indBuffer[threadIdx.x]]]) {

				currVal = -entropyConditionedOnRows(currDist);

				if(currVal > s_bestVal[threadIdx.x]){
					s_bestVal[threadIdx.x] = currVal;
					s_bestI[threadIdx.x] = threadIdx.x + window;
				}
			}

			// Restore current distribution
			currDist[0] -= s_weightBuffer[threadIdx.x*2];
			currDist[2] += s_weightBuffer[threadIdx.x*2];
			currDist[1] -= s_weightBuffer[threadIdx.x*2+1];
			currDist[3] += s_weightBuffer[threadIdx.x*2+1];

			if(syncThreads)
				__syncthreads();

			window += thread_group_size;
		}
	
		// Get split value
		if(threadIdx.x == 0){
			int bestI = 0;
			c_precision bestVal = s_bestVal[0];
			for(unsigned int i=1; i<blockDim.x; i++){
				if(s_bestVal[i] > bestVal){
					bestI = s_bestI[i];
					bestVal = s_bestVal[i];
				}
			}
			s_bestI[0] = bestI+1;
		}
	
		__syncthreads();

		// Assemble new distribution
		window = 0;
		unsigned int weight;
		while(threadIdx.x+window < s_bestI[0]){
			inst = params->nodeIndices[s_dataParams.nodeIndStart+threadIdx.x+window];
			//weight = bagWeights[s_dataParams.treeId*cb_constants.cb_instanceCount + inst];
			atomicAdd(&s_currDist[params->classValues[inst]],1);
			atomicAdd(&s_currDist[2+params->classValues[inst]],-1);
			window += thread_group_size;
		}
	}

	__device__ void evaluateNominalAttribute(paramPack_Kernel *params){
		unsigned int window = 0;
		unsigned int numInds = s_dataParams.numInds;
		unsigned int nodeIndStart = s_dataParams.nodeIndStart;
		unsigned int attribute = s_dataParams.attribute;
		unsigned int treeId = s_dataParams.treeId;
		unsigned int weight;
		unsigned int inst;
		unsigned int val;

		if(threadIdx.x < 40){
			s_currDist[threadIdx.x] = 0.0;
		}

		__syncthreads();

		// Split on median value
		while(threadIdx.x + window < numInds){
			inst = params->nodeIndices[nodeIndStart + threadIdx.x + window];
			c_precision datPoint = params->inputData[cb_constants.cb_instanceCount * attribute + params->inputInds[cb_constants.cb_instanceCount * attribute + inst]];
			if(datPoint >= 20)
				datPoint = 20;
			atomicAdd(&s_currDist[2*int(datPoint)+params->classValues[inst]],1);
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
		unsigned int nodes = (s_numValues >= max_nominal) ? 2 : s_numValues;

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
		unsigned int nodes = (s_numValues >= max_nominal) ? 2 : s_numValues;
		
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