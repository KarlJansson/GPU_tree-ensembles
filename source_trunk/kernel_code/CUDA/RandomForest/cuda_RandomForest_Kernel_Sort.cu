#include "cuda_RandomForest_Constants.cu"
#include "..\cuda_Common_Include.cu"
#include "..\cuda_Common_RNG.cu"

namespace Sort{
	__constant__ SharedBuffer cb_constants;

	// Host function for updates to constants
	__host__ void cuda_RandomForest_UpdateConstants(void* src){
		cudaMemcpyToSymbolAsync(cb_constants,src,sizeof(SharedBuffer));
	}

	struct sharedData{
		unsigned int nodeId;
		unsigned int seqStart;
		unsigned int numValues;
		unsigned int attOffset;

		unsigned int rndNum;
	};

	__device__ void nominalAttribute(paramPack_Kernel *params);
	__device__ void numericAttribute(paramPack_Kernel *params);

	__shared__ sharedData s_dataParams;
	__shared__ unsigned int s_numAttributeValues;

	__global__ void kernel_entry(paramPack_Kernel params){
		if(threadIdx.x == 0){
			s_dataParams.nodeId = params.innerNodeIds[cb_constants.cb_currentDepth+blockIdx.x+(cb_constants.cb_nodeBufferEnd+1)*cb_constants.cb_nodeIdFlip];
			s_dataParams.seqStart = params.nodeIndicesLimits[s_dataParams.nodeId];
			s_dataParams.numValues = params.nodeIndicesLimits[(cb_constants.cb_nodeBufferEnd+1)+s_dataParams.nodeId] - s_dataParams.seqStart + 1;

			stateRNG_xorShift128 state;
			state.w = params.stateBuffer[blockIdx.x];
			state.x = params.stateBuffer[(cb_constants.cb_nodeBufferEnd+1)+blockIdx.x];
			state.y = params.stateBuffer[(cb_constants.cb_nodeBufferEnd+1)*2+blockIdx.x];
			state.z = params.stateBuffer[(cb_constants.cb_nodeBufferEnd+1)*3+blockIdx.x];

			// Get random attribute to sort after
			s_dataParams.rndNum = xorShift128(&state) % cb_constants.cb_attributeCount;
			s_dataParams.attOffset = s_dataParams.rndNum*cb_constants.cb_instanceCount;

			s_numAttributeValues = params.attributeNumValues[s_dataParams.rndNum];
		}

		__syncthreads();

		// Skip sorting if attribute is nominal
		if(s_numAttributeValues >= max_nominal)
			numericAttribute(&params);
		else
			nominalAttribute(&params);
	}

	__device__ void numericAttribute(paramPack_Kernel *params){
		__shared__ int s_buckets[10];
		__shared__ unsigned int s_flagIntent[thread_group_size];

		unsigned int* writeBuffer = params->nodeIndicesMirror;
		unsigned int* readBuffer = params->nodeIndices;
		unsigned int window = 0;
		int bucket = 0;
		int exp = 1;
		int i;
		unsigned int nodeInd;
		int dataInd;

		// Radix sort
		while(cb_constants.cb_instanceCount / exp > 0){
			// Empty buckets
			if(threadIdx.x < 10)
				s_buckets[threadIdx.x] = 0;
			__syncthreads();

			// Count bucket sizes
			window = 0;
			while(threadIdx.x+window < s_dataParams.numValues){
				nodeInd = readBuffer[threadIdx.x + s_dataParams.seqStart + window];
				if(nodeInd > cb_constants.cb_instanceCount)
					nodeInd = 0;
				dataInd = params->inputInds[s_dataParams.attOffset+nodeInd];

				bucket = dataInd / exp % 10;
				atomicAdd(&s_buckets[bucket],1);
				window += thread_group_size;
			}
			__syncthreads();

			// Set bucket limits
			if(threadIdx.x == 0){
				int accum = 0;
				int tmp = 0;
				for(i=0; i<10; ++i){
					tmp = s_buckets[i];
					s_buckets[i] = accum;
					accum += tmp; 
				}
			}

			__syncthreads();

			//Write iteration result to memory
			window = 0;
			bool sync = 0;
			unsigned int turn;
			while(threadIdx.x+window < s_dataParams.numValues){
				sync = int(s_dataParams.numValues)-int(window) > 32 ? true : false;
				
				if(sync)
					__syncthreads();

				nodeInd = readBuffer[threadIdx.x + s_dataParams.seqStart + window];
				dataInd = params->inputInds[s_dataParams.attOffset+nodeInd];
				bucket = dataInd / exp % 10;
				
				// Communicate intent with the rest of the block
				s_flagIntent[threadIdx.x] = bucket;
				if(sync)
					__syncthreads();

				// Determin turn order
				turn = 0;
				for(unsigned int i=0; i<threadIdx.x; ++i){
					if(bucket == s_flagIntent[i])
						++turn;
				}

				// Write result at the threads alloted position
				writeBuffer[s_dataParams.seqStart+s_buckets[bucket]+turn] = nodeInd;

				if(sync)
					__syncthreads();

				// Prepare bucket offset for next iteration
				atomicAdd(&s_buckets[bucket],1);
				window += thread_group_size;
			}
			
			__syncthreads();

			unsigned int *tmp = writeBuffer;
			writeBuffer = readBuffer;
			readBuffer = tmp;
		
			// Advance digit position
			exp *= 10;
		}
	}

	__device__ void nominalAttribute(paramPack_Kernel *params){
		unsigned int window = 0;
		unsigned int nodeInd;
		while(threadIdx.x+window < s_dataParams.numValues){
			nodeInd = params->nodeIndices[threadIdx.x + s_dataParams.seqStart + window];
			params->nodeIndicesMirror[threadIdx.x + s_dataParams.seqStart + window] = nodeInd;
			window += thread_group_size;
		}
	}
}