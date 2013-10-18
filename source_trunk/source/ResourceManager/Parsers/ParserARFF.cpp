#include "stdafx.h"
#include "ParserARFF.h"

namespace DataMiner{
	DataDocumentPtr ParserARFF::parse(boost::filesystem::path path){
		beginParsing(path);

		int colId = 0;
		int classCol = 0;
		std::set<char> stopChars;

		// Parse attributes
		std::string attribute = "";
		std::string classAttribute = "";
		std::vector<std::string> attributes;
		std::vector<Attribute::InputFormat> formatVec;
		std::set<std::string> categoryStrings;

		stopChars.clear();
		stopChars.insert(' ');
		stopChars.insert('\n');
		stopChars.insert(',');
		stopChars.insert('}');
		// Attribute extraction
		while(true){
			// Remove comments
			while(m_buffer[m_readPosition] == '%')
				while(m_buffer[m_readPosition++] != '\n'){}

			if(m_buffer[m_readPosition] == '@'){
				attribute.clear();
				getToken(attribute,stopChars);

				if(attribute.compare("relation") == 0){
					attribute.clear();
					getToken(attribute,stopChars);
					classAttribute = attribute;
				}
				else if(attribute.compare("data") == 0){
					break;
				}
				else{
					attribute.clear();
					getToken(attribute,stopChars);

					if(attribute.compare("category") == 0 || attribute.compare("class") == 0 || attribute.compare("CLASS") == 0){
						attribute.clear();
						++m_readPosition;
						while(m_buffer[++m_readPosition] != '}'){
							if(m_buffer[m_readPosition] == ','){
								categoryStrings.insert(attribute);
								attribute.clear();
							}
							else
								attribute.push_back(m_buffer[m_readPosition]);
						}
						categoryStrings.insert(attribute);

						classCol = colId;
						attributes.push_back("class");
					}
					else{
						attributes.push_back(attribute);
						++colId;

						// Get format
						attribute.clear();
						getToken(attribute,stopChars);

						if(attribute.compare("numeric") == 0)
							formatVec.push_back(Attribute::IF_NUMERIC);
						else if(attribute.compare("nominal") == 0)
							formatVec.push_back(Attribute::IF_NOMINAL);
					}
				}
			}
			++m_readPosition;
		}

		for(unsigned int i=0; i<attributes.size(); ++i){
			if(i != classCol)
				m_document->addAttribute(AttributePtr(new Attribute(attributes[i])));
		}

		// Data extraction
		unsigned int attributeId = 0, attributeId2 = 0;
		Value::v_precision numberTest;
		std::stringstream sStream;

		char instanceBreakChar;
		if(m_buffer[m_readPosition+1] == '{'){
			instanceBreakChar = '}';
		}
		else
			instanceBreakChar = '\n';

		unsigned int cCount = 0;
		while(m_readPosition < m_size){
			attributeId = 0;
			attributeId2 = 0;
			colId = 0;
			++cCount;
			if(instanceBreakChar == '}')
				++m_readPosition;

			while(m_readPosition < m_size && (m_buffer[m_readPosition] != instanceBreakChar || colId == 0)){
				if(instanceBreakChar == '}'){
					attribute.clear();
					getToken(attribute,stopChars);

					sStream = std::stringstream(attribute);
					sStream >> attributeId;
					if(sStream.fail()){
						assert(0);
					}

					for(unsigned int i=attributeId2; i<attributeId; ++i){
						m_document->addAttributeValue(0.0,(colId<classCol?colId:colId-1));
						++colId;
					}
					attributeId2 = attributeId+1;
				}

				attribute.clear();
				getToken(attribute,stopChars);
				if(attribute.empty())
					break;

				sStream = std::stringstream(attribute);
				sStream >> numberTest;
				if(colId == classCol){
					if(sStream.fail()){
						m_document->addClassValue(attribute);
						categoryStrings.erase(attribute);
					}
					else
						m_document->addClassValue(numberTest);
				}
				else{
					if(sStream.fail())
						m_document->addAttributeValue(attribute,(colId<classCol?colId:colId-1));	
					else
						m_document->addAttributeValue(numberTest,(colId<classCol?colId:colId-1));
				}

				++colId;
			}

			if(instanceBreakChar == '}'){
				std::string categoryLeft = *categoryStrings.begin();
				for(unsigned int i=attributeId2; i<attributes.size(); ++i){
					if(colId == classCol)
						m_document->addClassValue(categoryLeft);
					else
						m_document->addAttributeValue(0.0,(colId<classCol+1?colId:colId-1));
					++colId;
				}
				++m_readPosition;
			}
		}

		endParsing();
		return m_document;
	}
}