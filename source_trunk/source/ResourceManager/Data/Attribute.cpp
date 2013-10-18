#include "stdafx.h"
#include "Attribute.h"

namespace DataMiner{
	Value::v_precision Attribute::getValue(unsigned int ind, bool replaceMissing, bool normalized){
		Value returnVal;
		if(!m_valueVector[ind]){
			if(replaceMissing)
				returnVal = m_missingSubstitute;
			else
				returnVal = -ValueMax;
		}
		else if(normalized){
			if(m_format == IF_NUMERIC)
				returnVal = *m_valueVector[ind];
			else
				returnVal = m_valueMap[m_valueVector[ind]->getValue()];

			double scale=2,translate=-1,value;
			if((m_max - m_min) != 0)
				returnVal = Value((((returnVal.getValue() - m_min) / (m_max - m_min)) * scale + translate));
		}
		else{
			if(m_format == IF_NUMERIC)
				returnVal = *m_valueVector[ind];
			else
				returnVal = m_valueMap[m_valueVector[ind]->getValue()];
		}
		return returnVal.getValue();
	}

	void Attribute::calculateMissingSubstitute(){
		unsigned int count = 0;
		std::map<Value::v_precision,unsigned int>::iterator itr = m_valueMap.begin();
		while(itr != m_valueMap.end()){
			m_missingSubstitute += itr->first;

			if(m_format == IF_NOMINAL && count == int(m_valueMap.size()/2)){
				m_missingSubstitute = itr->second;
				break;
			}

			itr++;
			++count;
		}

		if(m_format == IF_NUMERIC)
			m_missingSubstitute /= m_valueMap.size();
	}

	bool Attribute::missing(unsigned int ind){
		return (!m_valueVector[ind]) ? true : false; 
	}

	unsigned int Attribute::getNumMissing(){
		return m_numMissing;
	}

	void Attribute::insertValue(Value* value, int pos){
		if(value && m_valueMap.find(value->getValue()) == m_valueMap.end()){
			m_valueMap[value->getValue()] = m_numValues++;
		}
		
		if(m_valueMap.size() >= AttributeMaxNominal && m_format == IF_NOMINAL){
			m_format = IF_NUMERIC;
		}
		
		if(value && value->getValue() > m_max)
			m_max = value->getValue();
		if(value && value->getValue() < m_min)
			m_min = value->getValue();
		if(!value)
			++m_numMissing;

		if(pos == -1)
			m_valueVector.push_back(value);
		else
			m_valueVector[pos] = value;
	}
}