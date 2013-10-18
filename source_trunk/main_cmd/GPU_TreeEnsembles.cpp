// GPU_SupportVectorMachine_Unmanaged.cpp : Defines the entry point for the application.
//

#include "stdafx.h"
#include "GUIManagerCmd.h"
#include "DirectXManager.h"
#include "StartAlgorithmMessage.h"
#include "StopAlgorithmMessage.h"
#include "AlgorithmDataPack.h"
#include "MainFramework.h"
#include "ThreadManager.h"
#include "GUIGroup.h"
#include "HandlerStandard.h"

// Global Variables:
MainFrameworkPtr mainFramework = MainFrameworkPtr();

int _tmain(int argc, _TCHAR* argv[])
{
	std::wstring origin = argv[0];

	GUIManagerPtr gui = GUIManagerPtr(new DataMiner::GUIManagerCmd);
	
	IHandlerCmdPtr cmdHandler;
	cmdHandler = IHandlerCmdPtr(new HandlerStandard(gui));
	
	if(!cmdHandler->parseCommand(argc,argv)){
		return (int) 1;
	}

	// Create and initialize app framework
	DataMiner::ConfigManager::initialize();
	DataMiner::ThreadManager::initialize();
	mainFramework = MainFrameworkPtr(new DataMiner::MainFramework(DirectXManagerPtr(new DataMiner::DirectXManager()),gui));
	mainFramework->run();

	AlgorithmDataPackPtr data = AlgorithmDataPackPtr(new DataMiner::AlgorithmDataPack);
	data->m_algoName = gui->getEditText(IDC_COMBO_ALGO);
	data->m_dataResource = gui->getEditText(IDC_EDIT_FILEPATH);

	StartAlgorithmMessagePtr msg = StartAlgorithmMessagePtr(new DataMiner::StartAlgorithmMessage(data,1));
	mainFramework->postMessage(msg);
	msg->waitOnMessage();

	mainFramework.reset();
	DataMiner::ThreadManager::shutdown();
	return (int) 1;
}