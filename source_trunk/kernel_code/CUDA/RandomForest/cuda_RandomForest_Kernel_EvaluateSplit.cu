#include "cuda_RandomForest_Constants.cu"
#include "..\cuda_Common_Include.cu"
#include "..\cuda_Common_RNG.cu"

namespace EvaluateSplit{
	__constant__ SharedBuffer cb_constants;

	// Host function for updates to constants
	__host__ void cuda_RandomForest_UpdateConstants(void* src){
		cudaMemcpyToSymbolAsync(cb_constants,src,sizeof(SharedBuffer));
	}

	__device__ unsigned int sumEval(unsigned int *vec, int size);
	__device__ int maxIndexEval(unsigned int *vec, int size);

	// Entry point for kernel
	__global__ void kernel_entry(paramPack_Kernel params){
		unsigned int tid = blockIdx.x*thread_group_size+threadIdx.x;
		if(tid >= cb_constants.cb_availableNodes)
			return;

		unsigned int currentNode = params.innerNodeIds[tid + (cb_constants.cb_nodeBufferEnd+1)*cb_constants.cb_nodeIdFlip];
		int bestAttribute = params.attributeBuffer[currentNode];

		unsigned int nodeCounts[max_nominal];
		unsigned int sourceTree = params.treeIds[currentNode];
		unsigned int nodeIndStart = params.nodeIndicesLimits[currentNode];
		
		unsigned int numAttValues = params.attributeNumValues[bestAttribute];

		unsigned int dist[2];
		unsigned int newNodes;
		unsigned int newLeafs;

		bool leafs[max_nominal];
		unsigned int numNewNodes = 0;
		unsigned int numNewLeafs = 0;
		for(unsigned int i=0; i<((numAttValues >= max_nominal) ? 2 : numAttValues); ++i){
			dist[0] = params.distBuffer[currentNode+(cb_constants.cb_nodeBufferEnd+1)*(2*i)];
			dist[1] = params.distBuffer[currentNode+(cb_constants.cb_nodeBufferEnd+1)*(2*i+1)];
			nodeCounts[i] = dist[0] + dist[1];

			if(	(nodeCounts[i] < cb_constants.cb_maxInstInNodes) 
				|| ((dist[maxIndexEval(&dist[0],2)] - sumEval(&dist[0],2)) == 0)
				|| ((cb_constants.cb_maxDepth > 0)  &&  (cb_constants.cb_currentDepth+1 >= cb_constants.cb_maxDepth))){
				leafs[i] = true;
				++numNewLeafs;
			}
			else{
				leafs[i] = false;
				++numNewNodes;
			}
		}

		newNodes = atomicAdd(&params.check[0],numNewNodes+numNewLeafs);
		unsigned int innerNodeIdStart = atomicAdd(&params.check[2],numNewNodes);
		atomicAdd(&params.check[3],numNewLeafs);

		numNewNodes = 0;
		for(unsigned int i=0; i<((numAttValues >= max_nominal) ? 2 : numAttValues); ++i){
			dist[0] = params.distBuffer[currentNode+(cb_constants.cb_nodeBufferEnd+1)*(2*i)];
			dist[1] = params.distBuffer[currentNode+(cb_constants.cb_nodeBufferEnd+1)*(2*i+1)];

			if(leafs[i]){
				// Create leaf node
				params.attributeBuffer[newNodes+numNewNodes] = -1;
				params.treeIds[newNodes+numNewNodes] = sourceTree;
				params.classProbs[newNodes+numNewNodes] = c_precision(dist[0])/c_precision(nodeCounts[i]);
				params.classProbs[newNodes+numNewNodes+(cb_constants.cb_nodeBufferEnd+1)] = c_precision(dist[1])/c_precision(nodeCounts[i]);
				++numNewNodes;
			}
			else{
				// Create inner node
				params.attributeBuffer[newNodes+numNewNodes] = -2;
				params.treeIds[newNodes+numNewNodes] = sourceTree;

				params.classProbs[newNodes+numNewNodes] = c_precision(dist[0]);
				params.classProbs[newNodes+numNewNodes+(cb_constants.cb_nodeBufferEnd+1)] = c_precision(dist[1]);

				params.nodeIndicesLimits[newNodes+numNewNodes] = nodeIndStart;
				params.nodeIndicesLimits[newNodes+numNewNodes+(cb_constants.cb_nodeBufferEnd+1)] = nodeIndStart+nodeCounts[i]-1;

				unsigned int flip = ((cb_constants.cb_nodeIdFlip == 0) ? 1 : 0);
				params.innerNodeIds[innerNodeIdStart+(cb_constants.cb_nodeBufferEnd+1)*flip] = newNodes+numNewNodes;
				++innerNodeIdStart;
				++numNewNodes;
			}
			nodeIndStart += nodeCounts[i];
		}
		params.childIds[currentNode] = newNodes;
	}

	__device__ int maxIndexEval(unsigned int *vec, int size){
		unsigned int maximum = 0;
		int maxIndex = 0;

		for(unsigned int i = 0; i < size; i++){
			if((i == 0) || (vec[i] > maximum)){
				maxIndex = i;
				maximum = vec[i];
			}
		}

		return maxIndex;
	}

	__device__ unsigned int sumEval(unsigned int *vec, int size){
		unsigned int sum = 0;

		for(unsigned int i = 0; i < size; i++){
			sum += vec[i];
		}
		return sum;
	}
}