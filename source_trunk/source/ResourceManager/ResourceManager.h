#pragma once
#include "IParser.h"
#include "DataDocument.h"

namespace DataMiner{
	class ResourceManager{
	public:
		ResourceManager();

		void addParser(std::string name, IParserPtr parser);
		void unloadDocument(std::string doc);
		DataDocumentPtr parseDocument(boost::filesystem::path path);
		DataDocumentPtr getDocumentResource(boost::filesystem::path path);
		
		static boost::filesystem::path findFilePath(std::string filename, std::string dir = "");
		static std::string getResourcePath() { return m_resourcePath; }
		static std::vector<boost::filesystem::path> getFilesInFolder(std::string dir);
	private:
		std::map<std::string,IParserPtr> m_parsers;
		std::map<std::string,DataDocumentPtr> m_loadedDocuments;

		static std::string m_resourcePath;
	};
}

typedef boost::shared_ptr<DataMiner::ResourceManager> ResourceManagerPtr;