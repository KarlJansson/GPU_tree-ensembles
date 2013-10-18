#include "stdafx.h"
#include "SerialRandomTree.h"
#include "RandomForest.h"

namespace DataMiner{
	void SerialRandomTree::build(){
		// create the attribute indices window
		m_attIndicesWindow = std::vector<int>(data->numAttributes,0);
		int j = 0;
		for(int i = 0; i < m_attIndicesWindow.size(); i++){
			m_attIndicesWindow[i] = j++;
		}

		data->whatGoesWhere = std::vector<int>(data->inBag.size(),0);
		data->createInBagSortedIndices();

		int unprocessedNodes = 1;
		int newNodes = 0;
		
		SerialTreeNodePtr currentNode = SerialTreeNodePtr(new SerialTreeNode);
		currentNode->sortedIndices = (*data->sortedIndices);

		// compute initial class counts
		currentNode->classProbs = std::vector<double>(data->numClasses,0);
		for(int i = 0; i < data->numInstances; i++) {
			currentNode->classProbs[(*data->instClassValues)[i]] += (*data->instWeights)[i];
		}

		m_depth = 0;
		m_treeNodes.push_back(currentNode);
		while(unprocessedNodes != 0){
			// Iterate through unprocessed nodes
			while(unprocessedNodes != 0){
				currentNode = m_treeNodes[m_treeNodes.size()-(unprocessedNodes+newNodes)];

				// Check if node is a leaf
				if((currentNode->sortedIndices.size() > 0  &&  currentNode->sortedIndices[0].size() < max(2, getMinNum()))  // small
					|| (abs(currentNode->classProbs[maxIndex(currentNode->classProbs)] - sum(currentNode->classProbs)) < 1e-6)      // pure
					|| ((getMaxDepth() > 0)  &&  (m_depth >= getMaxDepth()))                           // deep
					){
					createLeafNode(currentNode);
					currentNode->clean();
				}
				else{
					if(findSplit(currentNode)){
						createSplitNodes(currentNode,newNodes);
						currentNode->clean();
					}
					else{
						createLeafNode(currentNode);
						currentNode->clean();
					}
				}
				unprocessedNodes--;
			}
			m_depth++;
			unprocessedNodes = newNodes;
			newNodes = 0;
		}

		m_nodesInTree = m_treeNodes.size();

		data.reset();
		m_dists.clear();
		m_splits.clear();
		m_props.clear();
		m_vals.clear();

		m_dists.shrink_to_fit();
		m_splits.shrink_to_fit();
		m_props.shrink_to_fit();
		m_vals.shrink_to_fit();
	}

	void SerialRandomTree::createLeafNode(SerialTreeNodePtr sourceNode){
		sourceNode->m_Attribute = -1;
		if(sourceNode->sortedIndices[0].size() != 0){
			for(size_t c = 0; c < sourceNode->classProbs.size(); c++){
				sourceNode->classProbs[c] /= sourceNode->sortedIndices[0].size();
			}
		}
		sourceNode->m_ClassProbs = sourceNode->classProbs;
	}

	bool SerialRandomTree::findSplit(SerialTreeNodePtr sourceNode){
		m_vals = std::vector<double>(data->numAttributes,0);
		m_dists = std::vector<std::vector<std::vector<double>>>(data->numAttributes,std::vector<std::vector<double>>());
		m_props = std::vector<std::vector<double>>(data->numAttributes,std::vector<double>());
		m_splits = std::vector<double>(data->numAttributes,0);

		// Investigate K random attributes
		int attIndex = 0;
		int windowSize = m_attIndicesWindow.size();
		int k = getKValue();
		bool sensibleSplitFound = false;
		double prior = 0;
		bool priorDone = false;
		boost::random::uniform_int_distribution<> indRand(0,windowSize-1);
		boost::random::uniform_int_distribution<> instRand(0,sourceNode->sortedIndices[0].size()-1);
		boost::random::mt19937 rng;
		rng.seed(m_seed);
		while((windowSize > 0) && (k-- > 0 || !sensibleSplitFound)){
			indRand = boost::random::uniform_int_distribution<>(0,windowSize-1);
			int chosenIndex = indRand(rng);//rand() % windowSize;//data.reusableRandomGenerator.nextInt(windowSize);

			attIndex = m_attIndicesWindow[chosenIndex];

			// shift chosen attIndex out of window
			m_attIndicesWindow[chosenIndex] = m_attIndicesWindow[windowSize - 1];
			m_attIndicesWindow[windowSize - 1] = attIndex;
			windowSize--;

			float split = 0;
			for(unsigned int i=0; i<10; ++i){
				split += (*data->vals)[attIndex][sourceNode->sortedIndices[/*attIndex*/0][instRand(rng)]];
			}
			split /= 10.0f;
			m_splits[attIndex] = split;//distribution(m_props, m_dists, attIndex, sourceNode->sortedIndices[attIndex]);

			// Calculate distribution
			int subSet, classVal,inst;
			m_dists[attIndex] = std::vector<std::vector<double>>(2,std::vector<double>(data->numClasses,0));
			for(unsigned int i=0; i<sourceNode->sortedIndices[0].size(); ++i){
				inst = sourceNode->sortedIndices[/*attIndex*/0][i];
				if(false && data->isAttrNominal(attIndex))
					subSet = (*data->vals)[attIndex][inst];
				else
					subSet = (*data->vals)[attIndex][inst] < split ? 0 : 1;
				classVal = (*data->instClassValues)[inst];
				m_dists[attIndex][subSet][classVal] += (*data->instWeights)[inst];
			}

			m_props[attIndex] = countsToFreqs(m_dists[attIndex]);

			if(!priorDone){ // needs to be computed only once per branch
				prior = entropyOverColumns(m_dists[attIndex]);
				priorDone = true;
			}
      
			double posterior = entropyConditionedOnRows(m_dists[attIndex]);
			m_vals[attIndex] = prior - posterior;  // we want the greatest reduction in entropy

			if(m_vals[attIndex] > 1e-2)   // we allow some leeway here to compensate
				sensibleSplitFound = true;   // for imprecision in entropy computation
      	}

		return sensibleSplitFound;
	}
	
	void SerialRandomTree::createSplitNodes(SerialTreeNodePtr sourceNode, int &newNodes){
		std::vector<SerialTreeNodePtr> splitNodes(2,SerialTreeNodePtr());

		sourceNode->m_Attribute = maxIndex(m_vals);   // find best attribute
		sourceNode->m_SplitPoint = m_splits[sourceNode->m_Attribute];
		sourceNode->m_Prop = m_props[sourceNode->m_Attribute];

		std::vector<std::vector<double>> chosenAttDists = m_dists[sourceNode->m_Attribute]; // remember dist for most important attribute
		m_dists.clear();
		m_splits.clear();
		m_props.clear();
		m_vals.clear();
      
		std::vector<std::vector<std::vector<int>>> subsetIndices(chosenAttDists.size(),std::vector<std::vector<int>>(data->numAttributes,std::vector<int>()));
		splitData(subsetIndices, sourceNode->m_Attribute, sourceNode->m_SplitPoint, sourceNode->sortedIndices);
      
		// Do not split if one branch is empty
		if(subsetIndices[0][0].size() == 0 || subsetIndices[1][0].size() == 0 ){
			createLeafNode(sourceNode);
			sourceNode->clean();
			return;
		}

		for(size_t i = 0; i < chosenAttDists.size(); i++){
			splitNodes[i] = SerialTreeNodePtr(new SerialTreeNode);

			// check if we're about to make an empty branch - this can happen with
			// nominal attributes with more than two categories (as of ver. 0.98)
			if(subsetIndices[i][0].size() == 0){
				for(size_t j = 0; j < chosenAttDists[i].size(); j++)
					chosenAttDists[i][j] = sourceNode->classProbs[j] / sourceNode->sortedIndices[0].size();
			}
			else{
				splitNodes[i]->m_Attribute = -1;
				splitNodes[i]->sortedIndices = subsetIndices[i];
				splitNodes[i]->classProbs = chosenAttDists[i];
				m_treeNodes.push_back(splitNodes[i]);
				newNodes++;
				sourceNode->m_children.push_back(splitNodes[i]);
			}
		}
		sourceNode->sortedIndices.clear();
	}

	std::vector<double> SerialRandomTree::distributionForInstance(InstancePtr instance){
		std::vector<double> result;
		SerialTreeNodePtr nodePtr = m_treeNodes[0];

		if(nodePtr->m_Attribute != -1 && instance->missing(nodePtr->m_Attribute)) {  // ---------------- missing value
			// TODO: Fix this function!
			result = std::vector<double>(m_MotherForest->m_document->getNumClassValues(),0);
			// split instance up
			for (int i = 0; i < m_Successors.size(); i++) {
				std::vector<double> help = m_Successors[i]->distributionForInstance(instance);
				for (int j = 0; j < help.size(); j++) {
					result[j] += m_Prop[i] * help[j];
				}
			}
		} 
		else if(false && nodePtr->m_Attribute != -1 && m_MotherForest->m_document->isNominalAttribute(nodePtr->m_Attribute)) { // ------ nominal
			while(nodePtr->m_Attribute != -1){
				nodePtr = nodePtr->m_children[(int) instance->getValue(nodePtr->m_Attribute)];
			}

			result = nodePtr->m_ClassProbs;
		} 
		else{ // ------------------------------------------ numeric attributes
			while(nodePtr->m_Attribute != -1){
				if(instance->getValue(nodePtr->m_Attribute) < nodePtr->m_SplitPoint) {
					nodePtr = nodePtr->m_children[0];
				}
				else{
					nodePtr = nodePtr->m_children[1];
				}
			}
			result = nodePtr->m_ClassProbs;
		}

		return result;
	}

	double SerialRandomTree::classifyInstance(InstancePtr instance){
		double result;

		return result;
	}
}