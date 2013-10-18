#include "stdafx.h"
#include "RandomTree.h"
#include "RandomForest.h"

namespace DataMiner{
	RandomTree::RandomTree(unsigned int seed):m_seed(seed){
		m_MinNum = 10;
	}

	inline float RandomTree::fast_log2(float val){
		int * const    exp_ptr = reinterpret_cast<int*>(&val);
		int            x = *exp_ptr;
		const int      log_2 = ((x >> 23) & 255) - 128;
		x &= ~(255 << 23);
		x += 127 << 23;
		*exp_ptr = x;

		val = ((-1.0f/3) * val + 2) * val - 2.0f/3;   // (1)

		return (val + log_2);
	}

	double RandomTree::lnFunc(double num){
		if(num <= 1e-6){
			return 0;
		} 
		else{
			return num * fast_log2(float(num));
			//return num * log(num);
		}
	}

	int RandomTree::maxIndex(std::vector<double> &vec){
		double maximum = 0;
		int maxIndex = 0;

		for(size_t i = 0; i < vec.size(); i++){
			if((i == 0) || (vec[i] > maximum)){
				maxIndex = i;
				maximum = vec[i];
			}
		}

		return maxIndex;
	}

	double RandomTree::sum(std::vector<double> &vec){
		double sum = 0;

		for(size_t i = 0; i < vec.size(); i++){
			sum += vec[i];
		}
		return sum;
	}

	std::vector<double> RandomTree::countsToFreqs(std::vector<std::vector<double>> &dist){
		std::vector<double> props(dist.size(),0);
    
		for(size_t k = 0; k < props.size(); k++){
			props[k] = sum(dist[k]);
		}
		if(sum(props) < 1e-6){
			for (size_t k = 0; k < props.size(); k++) {
				props[k] = 1.0 / (double) props.size();
			}
		} 
		else{
			normalize(props);
		}
		return props;
	}

	int RandomTree::getKValue(){
		return m_MotherForest->m_KValue;
	}

	int RandomTree::getMaxDepth() {
		return m_MotherForest->m_MaxDepth;
	}

	int RandomTree::getMinNum(){
		return m_MinNum;
	}

	double RandomTree::distribution(std::vector<std::vector<double>> &props, std::vector<std::vector<std::vector<double>>> &dists, int att, std::vector<int> &sortedIndices){
		double splitPoint = -DBL_MAX;
		std::vector<std::vector<double>> dist;
		size_t i;  
    
		if(false && data->isAttrNominal(att)){ // ====================== nominal attributes
			dist = std::vector<std::vector<double>>((*data->attNumVals)[att],std::vector<double>(data->numClasses,0));
			for (i = 0; i < sortedIndices.size(); i++) {
				int inst = sortedIndices[i];
				if ( data->isValueMissing(att,inst) )
					break;
				dist[ (int)(*data->vals)[att][inst] ][ (*data->instClassValues)[inst] ] += (*data->instWeights)[inst];        
			}

			splitPoint = 0; // signals we've found a sensible split point; by
						  // definition, a split on a nominal attribute is sensible
		} 
		else{ // ============================================ numeric attributes
			std::vector<std::vector<double>> currDist(2,std::vector<double>(data->numClasses,0));
			dist = std::vector<std::vector<double>>(2,std::vector<double>(data->numClasses,0));

			//begin with moving all instances into second subset
			for (size_t j = 0; j < sortedIndices.size(); j++) {
				int inst = sortedIndices[j];
				if ( data->isValueMissing(att,inst) ) 
					break;
				currDist[1][ (*data->instClassValues)[inst] ] += (*data->instWeights)[inst]; 
			}
      
			for (size_t j = 0; j < currDist.size(); j++)
				for(size_t k = 0; k < dist[j].size(); k++)
					dist[j][k] = currDist[j][k];
				//System.arraycopy(currDist[j], 0, dist[j], 0, dist[j].size());

			double currVal = -DBL_MAX; // current value of splitting criterion 
			double bestVal = -DBL_MAX; // best value of splitting criterion
			int bestI = 0; // the value of "i" BEFORE which the splitpoint is placed

			for (i = 1; i < sortedIndices.size(); i++) {  // --- try all split points

				int inst = sortedIndices[i];
				if ( data->isValueMissing(att,inst) ) 
					break;

				int prevInst = sortedIndices[i-1];

				currDist[0][ (*data->instClassValues)[prevInst]]
					+= (*data->instWeights)[prevInst];
				currDist[1][ (*data->instClassValues)[prevInst]]
					-= (*data->instWeights)[prevInst];        
        
				// do not allow splitting between two instances with the same value
				if((*data->vals)[att][inst] > (*data->vals)[att][prevInst]) {
					// we want the lowest impurity after split; at this point, we don't
					// really care what we've had before spliting
					currVal = -entropyConditionedOnRows(currDist);          
          
					if (currVal > bestVal) {
						bestVal = currVal;
						bestI = i;
					}
				}
			}                                             // ------- end split points

			/*
			 * Determine the best split point:
			 * bestI == 0 only if all instances had missing values, or there were
			 * less than 2 instances; splitPoint will remain set as -Double.MAX_VALUE. 
			 * This is not really a useful split, as all of the instances are 'below'
			 * the split line, but at least it's formally correct. And the dists[]
			 * also has a default value set previously.
			 */
			if ( bestI > 0 ) { // ...at least one valid splitpoint was found

				int instJustBeforeSplit = sortedIndices[bestI-1];
				int instJustAfterSplit = sortedIndices[bestI];
				splitPoint = ( (*data->vals)[att][instJustAfterSplit]
					+ (*data->vals)[att][instJustBeforeSplit] ) / 2.0;
        
				// Now make the correct dist[] from the default dist[] (all instances
				// in the second branch, by iterating through instances until we reach
				// bestI, and then stop.
				for ( int ii = 0; ii < bestI; ii++ ) {
					int inst = sortedIndices[ii];
					dist[0][ (*data->instClassValues)[inst] ] += (*data->instWeights)[inst];
					dist[1][ (*data->instClassValues)[inst] ] -= (*data->instWeights)[inst];                  
				}
			}
		} // ================================================== nominal or numeric?

		// compute total weights for each branch (= props)
		props[att] = countsToFreqs(dist);

		// distribute counts of instances with missing values

		// ver 0.96 - check for special case when *all* instances have missing vals
		if ( data->isValueMissing(att,sortedIndices[0]) )
			i = 0;

		while (i < sortedIndices.size()) {
			int inst = sortedIndices[i];
			for (size_t branch = 0; branch < dist.size(); branch++) {
				dist[ branch ][ (*data->instClassValues)[inst] ]
					+= props[ att ][ branch ] * (*data->instWeights)[inst];
			}
			i++;
		}

		// return distribution after split and best split point
		dists[att] = dist;
		return splitPoint;
	}
	
	void RandomTree::splitData(std::vector<std::vector<std::vector<int>>> &subsetIndices, int att, float splitPoint, std::vector<std::vector<int>> &sortedIndices){
		//Random random = data.reusableRandomGenerator;
		size_t j;
		std::vector<int> num; // how many instances go to each branch
		boost::random::uniform_real_distribution<double> realRand(0,data->numAttributes-1);
		boost::random::mt19937 rng;
		rng.seed(time(NULL));
		if(false && data->isAttrNominal(att)){ // ============================ if nominal
			num = std::vector<int>((*data->attNumVals)[att],0);

			for (j = 0; j < sortedIndices[/*att*/0].size(); j++) {
				int inst = sortedIndices[/*att*/0][j];

				if ( data->isValueMissing(att,inst) ) { // ---------- has missing value
					// decide where to put this instance randomly, with bigger branches
					// getting a higher chance
					double rn = realRand(rng);//random.nextDouble();
					int myBranch = -1;
					for (size_t k = 0; k < m_Prop.size(); k++) {
						rn -= m_Prop[k];
						if ( (rn <= 0) || k == (m_Prop.size()-1) ) {
							myBranch = k;
							break;
						}
					}

					data->whatGoesWhere[ inst ] = myBranch;
					num[myBranch]++;
				}
				else{ // ----------------------------- does not have missing value
					int subset = (int) (*data->vals)[att][inst];
					data->whatGoesWhere[ inst ] = subset;
					num[subset]++;
				} // --------------------------------------- end if has missing value
			}
		} 
		else{ // =================================================== if numeric
			num = std::vector<int>(2,0);

			for (j = 0; j < sortedIndices[/*att*/0].size(); j++){   
				int inst = sortedIndices[/*att*/0][j];
     
				//Instance inst = data.instance(sortedIndices[att][j]);

				if ( data->isValueMissing(att,inst) ) { // ---------- has missing value
					// for numeric attributes, ALWAYS num.length == 2
					// decide if instance goes into subset 0 or 1 randomly,
					// with bigger subsets having a greater probability of getting
					// the instance assigned to them
					// instances with missing values get processed LAST (sort order)
					// so branch sizes are known by now (and stored in m_Prop)
					double rn = realRand(rng); //random.nextDouble();
					int branch = ( rn > m_Prop[0] ) ? 1 : 0;
					data->whatGoesWhere[ inst ] = branch;
					num[ branch ]++;
				} 
				else{ // ----------------------------- does not have missing value
					int branch = ( (*data->vals)[att][inst] < splitPoint ) ? 0 : 1;
          
					data->whatGoesWhere[ inst ] = branch;
					num[ branch ]++;
				} // --------------------------------------- end if has missing value
			} // end for instance by instance
		}  // ============================================ end if nominal / numeric

		for(int a = 0; a < data->numAttributes; a++){
			/*if ( a == data->classIndex )
				continue;*/
      
			// create the new subset (branch) arrays of correct size
			for (size_t branch = 0; branch < num.size(); branch++) {
				subsetIndices[branch][a] = std::vector<int>(num[branch],0);
			}
		}
  
		for(int a = 0; a < data->numAttributes; a++){ // xxxxxxxxxx attr by attr
			/*if (a == data->classIndex)
				continue;*/
			for(size_t branch = 0; branch < num.size(); branch++){
				num[branch] = 0;
			}
      
			// fill them with stuff by looking at goesWhere array
			for(j = 0; j < sortedIndices[ /*a*/0 ].size(); j++){
				int inst = sortedIndices[ /*a*/0 ][j];
				int branch = data->whatGoesWhere[ inst ];
        
				subsetIndices[ branch ][ /*a*/0 ][ num[branch] ] = sortedIndices[/*a*/0][j];
				num[branch]++;
			}
		} // xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx end for attr by attr
	}

	void RandomTree::normalize(std::vector<double> &doubles){
		double sum = 0;
		for(size_t i = 0; i < doubles.size(); i++) {
			sum += doubles[i];
		}
		normalize(doubles, sum);
	}

	void RandomTree::normalize(std::vector<double> &doubles, double sum){
		if(sum == 0){
			return;
		}
		for(size_t i = 0; i < doubles.size(); i++){
			doubles[i] /= sum;
		}
	}

	double RandomTree::entropyConditionedOnRows(std::vector<std::vector<double>> &matrix){
		double returnValue = 0, sumForBranch;
		//double total = 0;

		for (size_t branchNum = 0; branchNum < matrix.size(); branchNum++) {
			sumForBranch = 0;
			for(size_t classNum = 0; classNum < matrix[0].size(); classNum++) {
				returnValue = returnValue + lnFunc(matrix[branchNum][classNum]);
				sumForBranch += matrix[branchNum][classNum];
			}
			returnValue = returnValue - lnFunc(sumForBranch);
			// total += sumForRow;
		}

		//return -returnValue / (total * log2);
		return -returnValue;
	}

	double RandomTree::entropyOverColumns(std::vector<std::vector<double>> &matrix){
		//return ContingencyTables.entropyOverColumns(matrix);
    
		double returnValue = 0, sumForColumn, total = 0;

		for (size_t j = 0; j < matrix[0].size(); j++) {
			sumForColumn = 0;
			for (size_t i = 0; i < matrix.size(); i++) {
				sumForColumn += matrix[i][j];
			}
			returnValue -= lnFunc(sumForColumn);
			total += sumForColumn;
		}

		//return (returnValue + lnFunc(total)) / (total * log2);
		return (returnValue + lnFunc(total)); 
	}
}