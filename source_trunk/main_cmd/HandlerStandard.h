#pragma once
#include "stdafx.h"
#include "IHandlerCmd.h"

class HandlerStandard : public IHandlerCmd{
public:
	HandlerStandard(GUIManagerPtr gui);
	bool parseCommand(int argc, _TCHAR* argv[]);
private:
};