#include "stdafx.h"
#include "ParserRaw.h"
#include "ResourceManager.h"

namespace DataMiner{
	DataDocumentPtr ParserRaw::parse(boost::filesystem::path path){
		beginParsing(path);

		unsigned int bufferPos = 0;
		int colId = 0;
		int classCol = 0;

		// Parse attributes
		std::string attribute;
		std::vector<std::string> attributes;
		while(m_buffer[bufferPos] != '\n'){
			while(m_buffer[bufferPos] != '\n' && m_buffer[bufferPos] != '	' && m_buffer[bufferPos] != ','){
				attribute += m_buffer[bufferPos];
				bufferPos++;
			}

			if(attribute.back() == '\r')
				attribute.pop_back();

			if(m_document->isClassAttribute(attribute))
				classCol = colId;
			
			attributes.push_back(attribute);

			attribute.clear();
			if(m_buffer[bufferPos] == '	' || m_buffer[bufferPos] == ',')
				++bufferPos;
			++colId;
		}
		++bufferPos;
		colId = 0;

		for(unsigned int i=0; i<attributes.size(); ++i){
			if(i != classCol)
				m_document->addAttribute(AttributePtr(new Attribute(attributes[i])));
		}

		// Parse data
		std::string dataString;
		Value::v_precision numberTest;
		while(bufferPos < m_size){
			dataString.clear();
			while(m_buffer[bufferPos] != '\n' && m_buffer[bufferPos] != ' ' && m_buffer[bufferPos] != '	'  && m_buffer[bufferPos] != ','  && bufferPos < m_size){
				if(m_buffer[bufferPos] != ' ')
					dataString += m_buffer[bufferPos];
				bufferPos++;
			}
			
			std::stringstream sStream(dataString);
			sStream >> numberTest;
			if(sStream.fail()){
				bool missingVal = false;
				if(m_document->isMissingValueMarker(sStream.str()))
					missingVal = true;

				// non-numeric value
				if(colId != classCol)
					m_document->addAttributeValue(sStream.str(),(colId<classCol?colId:colId-1),missingVal);
				else
					m_document->addClassValue(sStream.str());
			}
			else{
				// numeric value
				if(colId != classCol)
					m_document->addAttributeValue(numberTest,(colId<classCol?colId:colId-1));
				else
					m_document->addClassValue(numberTest);
			}
			
			if(m_buffer[bufferPos] == '\n'){
				assert(colId != m_document->getNumAttributes()+1);
				colId = 0;
			}
			else
				colId++;
			bufferPos++;
		}

		endParsing();
		return m_document;
	}
}