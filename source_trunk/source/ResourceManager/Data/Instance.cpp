#include "stdafx.h"
#include "Instance.h"
#include "DataDocument.h"

namespace DataMiner{
	Instance::Instance(Value* classValue, unsigned int size, unsigned int index):m_index(index),m_weight(1.0),m_classValue(classValue){
	}

	int Instance::classValue(){
		return int(m_classValue->getValue());
	}
		
	Value::v_precision Instance::weight(){
		return m_weight;
	}
		
	unsigned int Instance::numValues(){
		return m_motherDocument->getNumAttributes();
	}
		
	Value::v_precision Instance::getValue(unsigned int ind, bool replaceMissing, bool normalized){
		return m_motherDocument->getAttribute(ind)->getValue(m_index,replaceMissing,normalized);
	}
	
	bool Instance::missing(unsigned int ind){
		return m_motherDocument->getAttribute(ind)->missing(m_index);
	}

	unsigned int Instance::getIndex(){
		return m_index;
	}
}