#include "stdafx.h"
#include "IHandlerCmd.h"

void IHandlerCmd::printHelpText(){
	std::cerr << "Supported parameters:\n";

	std::string line;
	std::map<std::wstring,std::string>::iterator itr = m_supportedParameters.begin();
	while(itr != m_supportedParameters.end()){
		line.clear();
		line = std::string(itr->first.begin(),itr->first.end()) + "\n	" + itr->second + "\n";
		std::cout << line.c_str();
		itr++;
	}
}