#include "stdafx.h"
#include "DataDocument.h"

namespace DataMiner{
	DataDocument::DataDocument(){
		m_classValCount = 0;
		m_classFormat = Attribute::IF_NOMINAL;
		m_nominalMapCount = 0;

		m_missingValueMarkers.insert("MIS_VAL");
		m_missingValueMarkers.insert("?");

		m_classAttributeMarkers.insert("class");
		m_classAttributeMarkers.insert("ames");
	}

	InstancePtr DataDocument::getInstance(unsigned int index){
		if(index >= m_instances.size()){
			return InstancePtr();
		}

		return m_instances[index];
	}

	AttributePtr DataDocument::getAttribute(unsigned int index){
		if(index >= m_attributes.size()){
			return AttributePtr();
		}

		return m_attributes[index];
	}

	unsigned int DataDocument::getNumInstances(){
		return m_instances.size();
	}
	
	unsigned int DataDocument::getNumAttributes(){
		return m_attributes.size();
	}

	bool DataDocument::isNominalAttribute(int att){
		if(m_attributes[att]->getInputFormat() == Attribute::IF_NOMINAL)
			return true;
		return false;
	}

	bool DataDocument::isNumericAttribute(int att){
		if(m_attributes[att]->getInputFormat() == Attribute::IF_NUMERIC)
			return true;
		return false;
	}

	int DataDocument::nominalValue(std::string value, int attribute){
		std::map<std::string,std::map<int,int>>::iterator itr;

		if((itr = m_nominalMap.find(value)) == m_nominalMap.end() || itr->second.find(attribute) == itr->second.end()){
			m_nominalMap[value][attribute] = m_nominalCount[attribute]++;
			return m_nominalCount[attribute]-1;
		}
		else
			return itr->second[attribute];
	}

	void DataDocument::addAttribute(AttributePtr attribute){
		m_attributes.push_back(attribute);
	}

	void DataDocument::addAttributeValue(Value value, int attribute, bool missing){
		if(missing){
			m_missing.insert(m_attributeData.size());
			m_attributeData.push_back(MissingVal);
		}
		else
			m_attributeData.push_back(value);
	}

	void DataDocument::addAttributeValue(std::string value, int attribute, bool missing){
		if(isMissingValueMarker(value) || missing){
			m_missing.insert(m_attributeData.size());
			m_attributeData.push_back(MissingVal);
		}
		else{
			m_attributeData.push_back(nominalValue(value,attribute));
		}
	}

	bool DataDocument::isClassAttribute(std::string val){
		return m_classAttributeMarkers.find(val) != m_classAttributeMarkers.end();
	}
	
	bool DataDocument::isMissingValueMarker(std::string val){
		return m_missingValueMarkers.find(val) != m_missingValueMarkers.end();
	}

	void DataDocument::addClassValue(Value value){
		std::stringstream ss;
		ss << value.getValue();
		addClassValue(ss.str());
	}

	void DataDocument::addClassValue(std::string value){
		std::map<std::string,int>::iterator itr;
		m_classValues[value] = m_classValCount++;

		if((itr = m_classNominalMap.find(value)) == m_classNominalMap.end()){
			m_classNominalMap[value] = m_nominalMapCount++;
			m_classData.push_back(m_nominalMapCount-1);
		}
		else
			m_classData.push_back(itr->second);
	}

	void DataDocument::buildInstances(){
		int instId = 0;
		InstancePtr inst = InstancePtr(new Instance(&m_classData[instId++],m_attributes.size(),0));
		inst->m_motherDocument = shared_from_this();
		m_instances.reserve(m_attributeData.size()/m_attributes.size());
		assert(m_attributeData.size()%m_attributes.size() == 0);
		
		for(unsigned int i=0; i<m_attributeData.size(); i++){
			if(i != 0 && (i % m_attributes.size()) == 0){
				m_instances.push_back(inst);
				inst = InstancePtr(new Instance(&m_classData[instId++],m_attributes.size(),m_instances.size()));
				inst->m_motherDocument = shared_from_this();
			}

			if(m_missing.find(i) == m_missing.end()){
				m_attributes[i % m_attributes.size()]->insertValue(&m_attributeData[i]);
			}
			else{
				m_attributes[i % m_attributes.size()]->insertValue(NULL);
			}
		}
		m_instances.push_back(inst);

		for(unsigned int i=0; i<m_attributes.size(); ++i){
			m_attributes[i]->calculateMissingSubstitute();
		}

		std::map<int,unsigned int> classValues;
		for(unsigned int i=0; i<m_instances.size(); i++){
			classValues[m_instances[i]->classValue()]++;
		}

		std::map<int,unsigned int>::iterator clItr = classValues.begin();
	}

	void DataDocument::addMissingValueMarker(std::string val){
		m_missingValueMarkers.insert(val);
	}

	void DataDocument::addClassAttributeMarker(std::string val){
		m_classAttributeMarkers.insert(val);
	}
}