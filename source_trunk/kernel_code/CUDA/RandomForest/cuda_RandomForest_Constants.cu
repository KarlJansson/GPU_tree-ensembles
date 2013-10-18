#pragma once
typedef float c_precision;
#define thread_group_size 64
#define max_nominal 20

// Constant buffer strucs
struct SharedBuffer{
	unsigned int	cb_numTrees;
	unsigned int	cb_numFeatures;
	unsigned int	cb_maxDepth;
	unsigned int	cb_currentDepth;
	unsigned int	cb_availableNodes;
	unsigned int	cb_nodeBufferStart;
	unsigned int	cb_nodeBufferEnd;
	unsigned int	cb_maxInstInNodes;

	unsigned int	cb_instanceCount;
	unsigned int	cb_attributeCount;
	unsigned int	cb_nodeIdFlip;
};

struct ConstantBufferBagging{
	unsigned int	cb_treeOffset;
	unsigned int	cb_instanceCount;
	unsigned int	cb_nodeBufferEnd;
	bool			cb_baggingActivated;
};

struct ConstantBufferClassify{
	unsigned int	cb_numTrees;
	unsigned int	cb_treeOffset;
	unsigned int	cb_nodeBufferEnd;
	unsigned int	cb_majorityClass;

	unsigned int	cb_instanceCount;
	unsigned int	cb_attributeCount;
};

// Kernel parameter package structs
namespace ExtremeFindSplit{
	struct paramPack_Kernel{
		unsigned int	
			*treeIds,
			*rngStates,
			*nodeIndices,
			*distBuffer,
			*classValues,
			*bagWeights,
			*nodeIndicesLimits,
			*attributeTypes,
			*innerNodeIds,
			*childIds;
		int				
			*attributes;
		c_precision		
			*splitVals,
			*splitPoints,
			*dataset;
	};
}

namespace ExtremeMakeSplit{
	struct paramPack_Kernel{
		int				
			*attributes;
		unsigned int	
			*nodeIndices,
			*nodeIndicesMirror,
			*nodeIndicesLimits,
			*attributeTypes,
			*distBuffer,
			*innerNodeIds,
			*childIds,
			*classValues,
			*check;
		c_precision		
			*splitPoints,
			*dataset;
	};
}

namespace ExtremeCreateNodes{
	struct paramPack_Kernel{
		int				
			*attributes;
		unsigned int	
			*nodeIndices,
			*nodeIndicesMirror,
			*distBuffer,
			*childIds,
			*nodeIndicesLimits,
			*check,
			*attributeValCounts,
			*innerNodeIds,
			*treeIds;
		c_precision		
			*classProbs,
			*splitPoints;
	};
}

namespace Bagging{
	struct paramPack_Kernel{
		unsigned int	
			*stateBuffer,
			*treeIds,
			*bagWeights,
			*nodeIndicesLimits,
			*nodeIndices,
			*nodeIndicesMirror;
		int				
			*classValues;
		c_precision		
			*classProbs;
	};
}

namespace Sort{
	struct paramPack_Kernel{
		unsigned int	
			*nodeIndicesLimits,
			*nodeIndices,
			*nodeIndicesMirror,
			*inputInds,
			*stateBuffer,
			*attributeNumValues,
			*innerNodeIds;
	};
}

namespace FindSplit{
	struct paramPack_Kernel{
		unsigned int	
			*nodeIndicesLimits,
			*nodeIndices,
			*treeIds,
			*bagWeights,
			*distBuffer,
			*stateBuffer,
			*inputInds,
			*attributeNumValues,
			*innerNodeIds;
		int				
			*classValues,
			*attributeBuffer;
		c_precision		
			*splitPoints,
			*inputData,
			*splitVal,
			*classProbs;
	};
}

namespace EvaluateSplit{
	struct paramPack_Kernel{
		unsigned int	
			*inputInds,
			*bagWeights,
			*stateBuffer,
			*nodeIndicesLimits,
			*nodeIndices,
			*treeIds,
			*childIds,
			*distBuffer,
			*nodeIndicesMirror,
			*attributeNumValues,
			*innerNodeIds;
		int	
			*classValues, 
			*attributeBuffer,
			*check;
		c_precision 
			*splitPoints,
			*inputData,
			*classProbs;
	};
};

namespace RandomForest_Kernel_Classify{
	struct paramPack_Kernel{
		unsigned int	
			*attributeType,
			*childIds,
			*votes;
		int				
			*attributeBuffer;
		c_precision		
			*testData,
			*classProbs,
			*splitPoints;
	};
}

namespace RandomForest_SplitData_Kernel{
	struct paramPack_Kernel{
		unsigned int	
			*inputInds,
			*nodeIndicesLimits,
			*nodeIndices,
			*nodeIndicesMirror,
			*attributeNumValues,
			*innerNodeIds,
			*distBuffer;
		int 
			*attributeBuffer;
		c_precision
			*inputData,
			*splitPoints;
	};
}

namespace Build{
	struct paramPack_Kernel{
		unsigned int
			*inputInds,
			*bagWeights,
			*stateBuffer,
			*nodeIndicesLimits,
			*nodeIndices,
			*treeIds,
			*childIds;
		int
			*classValues, 
			*attributeBuffer,
			*check;
		c_precision 
			*splitPoints,
			*inputData,
			*classProbs;
	};
}

namespace KeplerBuild{
	struct paramPack_Kernel{
		unsigned int 
			*inputInds,
			*bagWeights,
			*stateBuffer,
			*nodeIndicesLimits,
			*nodeIndices,
			*nodeIndicesMirror,
			*treeIds,
			*childIds,
			*distBuffer;
		int 
			*classValues, 
			*attributeBuffer,
			*check;
		c_precision
			*splitVal,
			*inputData,
			*classProbs,
			*splitPoints;
	};
}