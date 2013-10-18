#include "stdafx.h"
#include "ParserRDS.h"

namespace DataMiner{
	DataDocumentPtr ParserRDS::parse(boost::filesystem::path path){
		beginParsing(path);

		unsigned int bufferPos = 0;
		unsigned int idCol = 0;
		unsigned int classCol = 0;
		unsigned int colCount = 0;

		// Parse input format
		std::string format;
		while(m_buffer[bufferPos] != '\n'){
			format.clear();
			while(m_buffer[bufferPos] != '\n' && m_buffer[bufferPos] != '	'){
				format += m_buffer[bufferPos];
				bufferPos++;
			}

			if(format.compare("class") == 0){
				classCol = colCount;
			}
			else if(format.compare("id") == 0){
				idCol = colCount;
			}

			if(m_buffer[bufferPos] == '	')
				bufferPos++;

			colCount++;
		}
		bufferPos++;

		colCount = 0;
		// Parse attributes
		while(m_buffer[bufferPos] != '\n'){
			std::string attribute = "";
			while(m_buffer[bufferPos] != '\n' && m_buffer[bufferPos] != '	'){
				attribute += m_buffer[bufferPos];
				bufferPos++;
			}

			if(colCount != classCol && colCount != idCol)
				m_document->addAttribute(AttributePtr(new Attribute(attribute)));

			if(m_buffer[bufferPos] == '	'){
				colCount++;
				bufferPos++;
			}
			else if(m_buffer[bufferPos] == '\n'){
				colCount = 0;
			}
		}
		bufferPos++;

		colCount = 0;

		// Parse data
		std::string dataString;
		Value::v_precision numberTest;
		unsigned int bufferPostPos;
		m_document->m_attributeData.reserve((m_size - bufferPos)/5);
		while(bufferPos < m_size){
			dataString.clear();
			while((m_buffer[bufferPos] == ' ' || m_buffer[bufferPos] == '	') && bufferPos < m_size){
				bufferPos++;
			}
			bufferPostPos = bufferPos;
			while(m_buffer[bufferPos] != '\n' && m_buffer[bufferPos] != '	' && m_buffer[bufferPos] != ' ' && bufferPos < m_size){
				bufferPos++;
			}

			dataString.assign(&m_buffer[bufferPostPos],bufferPos-bufferPostPos);

			if(colCount != idCol){
				try{
					numberTest = boost::lexical_cast<Value::v_precision,std::string>(dataString);
					if(colCount != classCol)
						m_document->addAttributeValue(numberTest,(colCount<idCol?colCount:colCount-1));
					else
						m_document->addClassValue(numberTest);
				}
				catch(boost::bad_lexical_cast &){
					if(colCount != classCol){
						bool missingVal = false;
						if(m_document->isMissingValueMarker(dataString))
							missingVal = true;

						m_document->addAttributeValue(dataString,(colCount<idCol?colCount:colCount-1),missingVal);
					}
					else{
						m_document->addClassValue(dataString);
					}
				}
			}

			if(m_buffer[bufferPos] == '\n'){
				colCount = 0;
			}
			else{
				colCount++;
			}
			
			bufferPos++;
		}

		endParsing();
		return m_document;
	}
}