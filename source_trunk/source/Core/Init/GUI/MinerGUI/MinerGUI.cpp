#include "stdafx.h"
#include "MinerGUI.h"
#include "GUIGroup.h"
#include "GUIManager.h"
#include <CommCtrl.h>

#include "ButtonRunAlgorithm.h"
#include "ButtonStopAlgorithm.h"
#include "ButtonLoadDataFile.h"
#include "ComboAlgorithmChoice.h"
#include "ButtonAddRunScheme.h"
#include "ButtonRunScheme.h"
#include "ButtonLoadScheme.h"
#include "ButtonRemoveScheme.h"

namespace DataMiner{
	MinerGUI::MinerGUI(GUIManagerPtr guiManager):m_guiManager(guiManager){
		// Add windows
		std::vector<std::wstring> items;
		m_mainGroup = GUIGroupPtr(new GUIGroup(guiManager,0,0));
		m_svmGroup = GUIGroupPtr(new GUIGroup(guiManager,1050,235));
		m_randForestGroup = GUIGroupPtr(new GUIGroup(guiManager,1050,235));
		m_algorithmGroup = GUIGroupPtr(new GUIGroup(guiManager,1050,35));
		m_evalMethodGroup = GUIGroupPtr(new GUIGroup(guiManager,1300,35));
		m_schemeGroup = GUIGroupPtr(new GUIGroup(guiManager,1050,435));

		m_guiManager->addGroup(IDC_GROUP_ALGO,m_algorithmGroup);
		m_guiManager->addGroup(IDC_GROUP_MAIN,m_mainGroup);
		m_guiManager->addGroup(IDC_GROUP_RANDFOREST,m_randForestGroup);
		m_guiManager->addGroup(IDC_GROUP_SVM,m_svmGroup);
		m_guiManager->addGroup(IDC_GROUP_EVAL,m_evalMethodGroup);
		m_guiManager->addGroup(IDC_GROUP_SCHEME,m_schemeGroup);

		// Scheme group
		m_schemeGroup->addWindow(m_guiManager->addWindow(IDC_SCHEME_BACKGROUND,L"BUTTON",0,WS_VISIBLE|WS_CHILD|BS_GROUPBOX,0,0,240,120,L"Scheme Manager:"));

		m_schemeGroup->addWindow(m_guiManager->addWindow(IDC_SCHEME_BUTTON_DELETE,L"BUTTON",0,WS_TABSTOP|WS_VISIBLE|WS_CHILD|BS_DEFPUSHBUTTON,10,80,105,25,L"Remove Run"));
		m_guiManager->getWindow(IDC_SCHEME_BUTTON_DELETE)->setOnClickFunction(RunnablePtr(new ButtonRemoveScheme));
		m_schemeGroup->addWindow(m_guiManager->addWindow(IDC_SCHEME_BUTTON_ADD,L"BUTTON",0,WS_TABSTOP|WS_VISIBLE|WS_CHILD|BS_DEFPUSHBUTTON,10,50,105,25,L"Add Run"));
		m_guiManager->getWindow(IDC_SCHEME_BUTTON_ADD)->setOnClickFunction(RunnablePtr(new ButtonAddRunScheme));
		m_schemeGroup->addWindow(m_guiManager->addWindow(IDC_SCHEME_BUTTON_LOAD,L"BUTTON",0,WS_TABSTOP|WS_VISIBLE|WS_CHILD|BS_DEFPUSHBUTTON,125,50,105,25,L"Load Scheme"));
		m_guiManager->getWindow(IDC_SCHEME_BUTTON_LOAD)->setOnClickFunction(RunnablePtr(new ButtonLoadScheme));
		m_schemeGroup->addWindow(m_guiManager->addWindow(IDC_SCHEME_BUTTON_RUN,L"BUTTON",0,WS_TABSTOP|WS_VISIBLE|WS_CHILD|BS_DEFPUSHBUTTON,125,80,105,25,L"Run Scheme"));
		m_guiManager->getWindow(IDC_SCHEME_BUTTON_RUN)->setOnClickFunction(RunnablePtr(new ButtonRunScheme));
		m_schemeGroup->addWindow(m_guiManager->addWindow(IDC_SCHEME_COMBO_ITEMS,L"COMBOBOX",WS_EX_CLIENTEDGE,WS_VISIBLE|WS_CHILD|LBS_STANDARD,10,20,220,21,L""));

		// Main group
		m_mainGroup->addWindow(m_guiManager->addWindow(IDC_BUTTON_RUN,L"BUTTON",0,WS_TABSTOP|WS_VISIBLE|WS_CHILD|BS_DEFPUSHBUTTON,10,5,105,25,L"Run"));
		m_guiManager->getWindow(IDC_BUTTON_RUN)->setOnClickFunction(RunnablePtr(new ButtonRunAlgorithm));

		m_mainGroup->addWindow(m_guiManager->addWindow(IDC_BUTTON_STOP,L"BUTTON",0,WS_TABSTOP|WS_VISIBLE|WS_CHILD|BS_DEFPUSHBUTTON,120,5,105,25,L"Stop"));
		m_guiManager->getWindow(IDC_BUTTON_STOP)->setOnClickFunction(RunnablePtr(new ButtonStopAlgorithm));

		m_mainGroup->addWindow(m_guiManager->addWindow(IDC_BUTTON_LOAD,L"BUTTON",0,WS_TABSTOP|WS_VISIBLE|WS_CHILD|BS_DEFPUSHBUTTON,345,5,105,25,L"Load"));
		m_guiManager->getWindow(IDC_BUTTON_LOAD)->setOnClickFunction(RunnablePtr(new ButtonLoadDataFile));

		m_guiManager->getWindow(IDC_BUTTON_STOP)->disable();

		m_mainGroup->addWindow(m_guiManager->addWindow(IDC_PROGRESSBAR_PROGRESS,PROGRESS_CLASS,0,WS_VISIBLE|WS_CHILD|PBS_SMOOTH,230,5,110,15,L""));
		m_mainGroup->addWindow(m_guiManager->addWindow(IDC_PROGRESSBAR_PROGRESS2,PROGRESS_CLASS,0,WS_VISIBLE|WS_CHILD|PBS_SMOOTH,230,21,110,8,L""));
		
		m_mainGroup->addWindow(m_guiManager->addWindow(IDC_EDIT_FILEPATH,L"EDIT",WS_EX_CLIENTEDGE,WS_VISIBLE|WS_CHILD,455,7,580,21,L"..\\..\\..\\DataSets\\Mushroom.txt"));
		m_mainGroup->addWindow(m_guiManager->addWindow(IDC_STATIC_INFOTEXT,L"EDIT",0,ES_MULTILINE|WS_VSCROLL|WS_BORDER|WS_VISIBLE|WS_CHILD,10,35,505,740,L""));
		m_mainGroup->addWindow(m_guiManager->addWindow(IDC_STATIC_DEBUG,L"EDIT",0,ES_MULTILINE|WS_VSCROLL|WS_BORDER|WS_VISIBLE|WS_CHILD,530,35,505,740,L""));
		
		// Algorithm group
		m_algorithmGroup->addWindow(m_guiManager->addWindow(IDC_ALGORITHM_BACKGROUND,L"BUTTON",0,WS_VISIBLE|WS_CHILD|BS_GROUPBOX,0,0,240,190,L"Algorithm settings:"));

		m_algorithmGroup->addWindow(m_guiManager->addWindow(IDC_STATIC_ALGORITHMCHOICETEXT,L"STATIC",0,WS_VISIBLE|WS_CHILD,10,25,100,20,L"Algorithm type:"));
		m_algorithmGroup->addWindow(m_guiManager->addWindow(IDC_STATIC_EVALUATIONCHOICETEXT,L"STATIC",0,WS_VISIBLE|WS_CHILD,10,55,100,20,L"Evaluation method:"));
		m_algorithmGroup->addWindow(m_guiManager->addWindow(IDC_STATIC_GPUAPITEXT,L"STATIC",0,WS_VISIBLE|WS_CHILD,10,85,100,20,L"GPGPU API:"));
		
		m_algorithmGroup->addWindow(m_guiManager->addWindow(IDC_EDIT_EVALPARAM,L"EDIT",WS_EX_CLIENTEDGE,WS_VISIBLE|WS_CHILD,110,115,40,21,L"70"));
		//m_algorithmGroup->addWindow(m_guiManager->addWindow(IDC_EDIT_GRIDSTART,L"EDIT",WS_EX_CLIENTEDGE,WS_VISIBLE|WS_CHILD,1170,192,40,21,L"0"));
		//m_algorithmGroup->addWindow(m_guiManager->addWindow(IDC_EDIT_PERFSTART,L"EDIT",WS_EX_CLIENTEDGE,WS_VISIBLE|WS_CHILD,1170,162,40,21,L"3"));
		//m_algorithmGroup->addWindow(m_guiManager->addWindow(IDC_BUTTON_GRIDSEARCH,L"BUTTON",0,WS_TABSTOP|WS_VISIBLE|WS_CHILD|BS_DEFPUSHBUTTON,1060,190,105,25,L"Grid search"));
		//m_algorithmGroup->addWindow(m_guiManager->addWindow(IDC_BUTTON_PERFSEARCH,L"BUTTON",0,WS_TABSTOP|WS_VISIBLE|WS_CHILD|BS_DEFPUSHBUTTON,1060,160,105,25,L"Perf search"));

		m_algorithmGroup->addWindow(m_guiManager->addWindow(IDC_COMBO_EVALUATION,L"COMBOBOX",WS_EX_CLIENTEDGE,WS_VISIBLE|WS_CHILD|LBS_STANDARD,110,50,120,21,L""));
		items.push_back(L"PercentageSplit");
		items.push_back(L"CrossValidation");
		m_guiManager->getWindow(IDC_COMBO_EVALUATION)->addItemsToWindow(items);
		items.clear();

		m_algorithmGroup->addWindow(m_guiManager->addWindow(IDC_ALGORITHM_GPUAPI,L"COMBOBOX",WS_EX_CLIENTEDGE,WS_VISIBLE|WS_CHILD|LBS_STANDARD,110,80,65,21,L""));
		items.push_back(L"CUDA");
		items.push_back(L"DirectX");
		items.push_back(L"OpenCL");
		m_guiManager->getWindow(IDC_ALGORITHM_GPUAPI)->addItemsToWindow(items);
		items.clear();
		
		m_algorithmGroup->addWindow(m_guiManager->addWindow(IDC_COMBO_ALGO,L"COMBOBOX",WS_EX_CLIENTEDGE,WS_VISIBLE|WS_CHILD|LBS_STANDARD,110,20,120,21,L""));
		items.push_back(L"GPURandomForest");
		items.push_back(L"CPURandomForest");
		items.push_back(L"GPUSVM");
		items.push_back(L"CPUSVM");
		m_guiManager->getWindow(IDC_COMBO_ALGO)->addItemsToWindow(items);
		m_guiManager->getWindow(IDC_COMBO_ALGO)->setOnClickFunction(RunnablePtr(new ComboAlgorithmChoice));
		items.clear();

		// Evaluation method group
		//m_evalMethodGroup->addWindow(m_guiManager->addWindow(IDC_EVALMETHOD_BACKGROUND,L"BUTTON",0,WS_VISIBLE|WS_CHILD|BS_GROUPBOX,0,0,200,190,L"Evaluation method settings:"));

		// Random forest group
		m_randForestGroup->addWindow(m_guiManager->addWindow(IDC_RANDFOREST_BACKGROUND,L"BUTTON",0,WS_VISIBLE|WS_CHILD|BS_GROUPBOX,0,0,240,195,L"Random forest settings:"));

		int startHeight = 30;
		m_randForestGroup->addWindow(m_guiManager->addWindow(IDC_RANDFOREST_NUMTREESTEXT,L"STATIC",0,WS_VISIBLE|WS_CHILD,
			10,startHeight+20,95,20,L"Number of trees:"));
		m_randForestGroup->addWindow(m_guiManager->addWindow(IDC_RANDFOREST_TREEDEPTHTEXT,L"STATIC",0,WS_VISIBLE|WS_CHILD,
			10,startHeight+45,95,20,L"Tree depth:"));
		m_randForestGroup->addWindow(m_guiManager->addWindow(IDC_RANDFOREST_NUMFEATURESTEXT,L"STATIC",0,WS_VISIBLE|WS_CHILD,
			10,startHeight+70,95,20,L"Number of features:"));
		m_randForestGroup->addWindow(m_guiManager->addWindow(IDC_RANDFOREST_SEEDTEXT,L"STATIC",0,WS_VISIBLE|WS_CHILD,
			10,startHeight+95,95,20,L"Seed:"));
		m_randForestGroup->addWindow(m_guiManager->addWindow(IDC_RANDFOREST_MAXINSTTEXT,L"STATIC",0,WS_VISIBLE|WS_CHILD,
			10,startHeight+120,130,20,L"Max instances per node:"));

		m_randForestGroup->addWindow(m_guiManager->addWindow(IDC_RANDFOREST_NUMTREES,L"EDIT",WS_EX_CLIENTEDGE,WS_VISIBLE|WS_CHILD,
			180,startHeight+20,50,20,L"1"));
		m_randForestGroup->addWindow(m_guiManager->addWindow(IDC_RANDFOREST_TREEDEPTH,L"EDIT",WS_EX_CLIENTEDGE,WS_VISIBLE|WS_CHILD,
			180,startHeight+45,50,20,L"100"));
		m_randForestGroup->addWindow(m_guiManager->addWindow(IDC_RANDFOREST_NUMFEATURES,L"EDIT",WS_EX_CLIENTEDGE,WS_VISIBLE|WS_CHILD,
			180,startHeight+70,50,20,L"5"));
		m_randForestGroup->addWindow(m_guiManager->addWindow(IDC_RANDFOREST_SEED,L"EDIT",WS_EX_CLIENTEDGE,WS_VISIBLE|WS_CHILD,
			180,startHeight+95,50,20,L"1"));
		m_randForestGroup->addWindow(m_guiManager->addWindow(IDC_RANDFOREST_MAXINST,L"EDIT",WS_EX_CLIENTEDGE,WS_VISIBLE|WS_CHILD,
			180,startHeight+120,50,20,L"10"));

		m_randForestGroup->addWindow(m_guiManager->addWindow(IDC_RANDFOREST_ITSELECTOR,L"COMBOBOX",WS_EX_CLIENTEDGE,WS_VISIBLE|WS_CHILD|LBS_STANDARD,
			10,20,120,21,L""));
		items.push_back(L"Iteration_4");
		items.push_back(L"Iteration_2");
		items.push_back(L"Iteration_3");
		items.push_back(L"Iteration_1");
		m_guiManager->getWindow(IDC_RANDFOREST_ITSELECTOR)->addItemsToWindow(items);
		items.clear();

		// SVM group
		m_svmGroup->addWindow(m_guiManager->addWindow(IDC_SVM_BACKGROUND,L"BUTTON",0,WS_VISIBLE|WS_CHILD|BS_GROUPBOX,0,0,240,110,L"SVM settings:"));
		m_svmGroup->addWindow(m_guiManager->addWindow(IDC_EDIT_PARAM2,L"EDIT",WS_EX_CLIENTEDGE,WS_VISIBLE|WS_CHILD,65,80,50,21,L"0.125"));
		m_svmGroup->addWindow(m_guiManager->addWindow(IDC_EDIT_PARAM3,L"EDIT",WS_EX_CLIENTEDGE,WS_VISIBLE|WS_CHILD,120,80,50,21,L"1.0"));
		m_svmGroup->addWindow(m_guiManager->addWindow(IDC_EDIT_C,L"EDIT",WS_EX_CLIENTEDGE,WS_VISIBLE|WS_CHILD,10,50,100,21,L"1.0"));

		m_svmGroup->addWindow(m_guiManager->addWindow(IDC_COMBO_KERNEL,L"COMBOBOX",WS_EX_CLIENTEDGE,WS_VISIBLE|WS_CHILD|LBS_STANDARD,10,80,50,21,L""));
		items.push_back(L"RBF");
		items.push_back(L"Puk");
		m_guiManager->getWindow(IDC_COMBO_KERNEL)->addItemsToWindow(items);
		items.clear();

		m_svmGroup->addWindow(m_guiManager->addWindow(IDC_COMBO_KERNELCACHE,L"COMBOBOX",WS_EX_CLIENTEDGE,WS_VISIBLE|WS_CHILD|LBS_STANDARD,10,22,50,21,L""));
		items.push_back(L"true");
		items.push_back(L"false");
		m_guiManager->getWindow(IDC_COMBO_KERNELCACHE)->addItemsToWindow(items);
		items.clear();

		m_svmGroup->addWindow(m_guiManager->addWindow(IDC_COMBO_KERNELCACHEFULL,L"COMBOBOX",WS_EX_CLIENTEDGE,WS_VISIBLE|WS_CHILD|LBS_STANDARD,65,22,50,21,L""));
		items.push_back(L"false");
		items.push_back(L"true");
		m_guiManager->getWindow(IDC_COMBO_KERNELCACHEFULL)->addItemsToWindow(items);
		items.clear();

		m_svmGroup->hide();
	}

	MinerGUI::~MinerGUI(){
		
	}

	void MinerGUI::disableAllButStop(){
		m_mainGroup->disable();
		m_svmGroup->disable();
		m_randForestGroup->disable();
		m_evalMethodGroup->disable();
		m_schemeGroup->disable();
		m_guiManager->getWindow(IDC_BUTTON_STOP)->enable();
		m_guiManager->setProgressBar(IDC_PROGRESSBAR_PROGRESS,100,0);
		m_guiManager->setProgressBar(IDC_PROGRESSBAR_PROGRESS2,100,0);
	}

	void MinerGUI::enableAllButStop(){
		m_mainGroup->enable();
		m_svmGroup->enable();
		m_randForestGroup->enable();
		m_evalMethodGroup->enable();
		m_schemeGroup->enable();
		m_guiManager->getWindow(IDC_BUTTON_STOP)->disable();
		m_guiManager->setProgressBar(IDC_PROGRESSBAR_PROGRESS,100,0);
		m_guiManager->setProgressBar(IDC_PROGRESSBAR_PROGRESS2,100,0);
	}
}