#pragma once
#include "stdafx.h"

class IHandlerCmd{
public:
	IHandlerCmd(GUIManagerPtr gui):m_gui(gui){}
	virtual bool parseCommand(int argc, _TCHAR* argv[]) = 0;
protected:
	void printHelpText();

	std::map<std::wstring,std::string> m_supportedParameters;
	std::map<std::wstring,std::wstring> m_parameterMap;
	GUIManagerPtr m_gui;
};

typedef boost::shared_ptr<IHandlerCmd> IHandlerCmdPtr;