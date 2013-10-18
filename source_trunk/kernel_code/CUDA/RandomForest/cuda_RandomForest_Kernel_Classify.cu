#include "cuda_RandomForest_Constants.cu"
#include "..\cuda_Common_Include.cu"

namespace RandomForest_Kernel_Classify{
	__constant__ ConstantBufferClassify cb_constants;

	// Host function for updates to constants
	__host__ void cuda_RandomForest_UpdateConstants(void* src){
		cudaMemcpyToSymbol(cb_constants,src,sizeof(ConstantBufferClassify));
	}

	__global__ void kernel_entry(paramPack_Kernel params){
		unsigned int tid = cb_constants.cb_treeOffset + threadIdx.x + blockIdx.x*thread_group_size;
		if(tid >= cb_constants.cb_numTrees*cb_constants.cb_instanceCount)
			return;

		// Access pattern
		// instances*trees: 5*5 = 25
		// tid:				0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
		// instances:		0 1 2 3 4 0 1 2 3 4 0  1  2  3  4  0  1  2  3  4  0  1  2  3  4
		// trees:			0 0 0 0 0 1 1 1 1 1 2  2  2  2  2  3  3  3  3  3  4  4  4  4  4

		unsigned int classSelector = 0;
		unsigned int pos;
		unsigned int instanceId = tid%cb_constants.cb_instanceCount;
		unsigned int treeId = c_precision(tid)/c_precision(cb_constants.cb_instanceCount);

		c_precision splitPoint, dataPoint;
		int attribute;
		unsigned int attType;

		// Traverse tree for instances
		pos = treeId;
		unsigned int parent;
		while((attribute = params.attributeBuffer[pos]) != -1){
			parent = pos;
			attType = params.attributeType[attribute];
			if(attType > max_nominal){
				splitPoint = params.splitPoints[pos];
				dataPoint = params.testData[instanceId*cb_constants.cb_attributeCount + attribute];

				if(dataPoint != -FLT_MAX)
					pos = (dataPoint < splitPoint) ? params.childIds[pos] : params.childIds[pos]+1;
				else
					pos = params.childIds[pos];
			}
			else{
				dataPoint = params.testData[instanceId*cb_constants.cb_attributeCount + attribute];
				if(dataPoint != -FLT_MAX)
					pos = int(dataPoint) + params.childIds[pos];
				else
					pos = params.childIds[pos];
			}
		}

		// Decide class of instance
		c_precision cl1 = params.classProbs[pos];
		c_precision cl2 = params.classProbs[pos+(cb_constants.cb_nodeBufferEnd+1)];

		if(abs(cl1-cl2) < 1.0e-4){
			cl1 = params.classProbs[parent];
			cl2 = params.classProbs[parent+(cb_constants.cb_nodeBufferEnd+1)];
			c_precision numInst = cl1+cl2;
			cl1 /= numInst;
			cl2 /= numInst;

			if(cl1 > cl2)
				atomicAdd(&params.votes[2*instanceId],1);
			else
				atomicAdd(&params.votes[2*instanceId+1],1);
		}
		else{
			if(cl1 > cl2)
				atomicAdd(&params.votes[2*instanceId],1);
			else
				atomicAdd(&params.votes[2*instanceId+1],1);
		}
	}
}