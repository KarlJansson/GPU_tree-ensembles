#include "stdafx.h"
#include "HandlerStandard.h"
#include "GUIManagerCmd.h"

HandlerStandard::HandlerStandard(GUIManagerPtr gui):IHandlerCmd(gui){
	m_supportedParameters[L"d"] = "Dataset path.";
	m_supportedParameters[L"a"] = "Defines the algorithm used; GPURF, GPUERT or CPUERT.\n	Default: CPUERT";
	m_supportedParameters[L"e"] = "Evaluation method used; cValidation or pSplit.\n	Default: cValidation";
	m_supportedParameters[L"pE"] = "Numeric evaluation parameter.\n	Number of folds or the percentage used for training.\n	Default: 10/70";
	m_supportedParameters[L"pTrees"] = "Numeric parameter value for random forest algorithms.\n	The number of trees in the forest.\n	Default: 100";
	m_supportedParameters[L"pMIPN"] = "Numeric parameter value for random forest algorithms.\n	The maximum number of nodes in a node before it turns into a leaf.\n	Default: 10";
	m_supportedParameters[L"pDepth"] = "Numeric parameter value for random forest algorithms.\n	The maximum depth of the forest.\n	Default: 100";
	m_supportedParameters[L"pK"] = "Numeric parameter value for random forest algorithms.\n	The number of features considered in each split.\n	Default: log(numFeatures)+1";
	m_supportedParameters[L"pSeed"] = "Numeric parameter value for random forest algorithms.\n	The seed of the random number generators.\n	Default: 1";
	m_supportedParameters[L"pThreads"] = "Number of threads/GPUs that will be used.\n	Default: All";
}

bool HandlerStandard::parseCommand(int argc, _TCHAR* argv[]){
	if(argc < 2){
		printHelpText();
		return false;
	}

	for(unsigned int i=1; i<argc; i+=2){
		if(m_supportedParameters.find(argv[i]) != m_supportedParameters.end() && i+1 <= argc)
			m_parameterMap[argv[i]] = argv[i+1];
		else{
			std::cerr << "Invalid parameter syntax used!\n\n\n";
			printHelpText();
			return false;
		}
	}

	std::map<std::wstring,std::wstring>::iterator itr;

	if((itr = m_parameterMap.find(L"d")) != m_parameterMap.end())
		m_gui->setText(IDC_EDIT_FILEPATH,itr->second);
	else
		return false;

	if((itr = m_parameterMap.find(L"pThreads")) != m_parameterMap.end())
		DataMiner::ConfigManager::setSetting("pThreads",std::string(itr->second.begin(),itr->second.end()));
	else
		DataMiner::ConfigManager::setSetting("pThreads","All");

	// Algorithm
	if((itr = m_parameterMap.find(L"a")) != m_parameterMap.end()){
		if(itr->second.compare(L"GPURF") == 0)
			m_gui->setText(IDC_RANDFOREST_ITSELECTOR,L"Iteration_2");
		else if(itr->second.compare(L"GPUERT") == 0)
			m_gui->setText(IDC_RANDFOREST_ITSELECTOR,L"Iteration_4");

		if(itr->second.find(L"GPU") != std::string::npos)
			m_gui->setText(IDC_COMBO_ALGO,L"GPURandomForest");
		else if(itr->second.find(L"CPU") != std::string::npos)
			m_gui->setText(IDC_COMBO_ALGO,L"CPURandomForest");
	}
	else{
		m_gui->setText(IDC_COMBO_ALGO,L"CPURandomForest");
	}
	
	// Evaluation
	if((itr = m_parameterMap.find(L"e")) != m_parameterMap.end()){
		if(itr->second.compare(L"cValidation") == 0){
			m_gui->setText(IDC_COMBO_EVALUATION,L"CrossValidation");
			if((itr = m_parameterMap.find(L"pE")) != m_parameterMap.end()){
				m_gui->setText(IDC_EDIT_EVALPARAM,itr->second);
			}
			else
				m_gui->setText(IDC_EDIT_EVALPARAM,L"10");
		}
		else if(itr->second.compare(L"pSplit") == 0){
			m_gui->setText(IDC_COMBO_EVALUATION,L"PercentageSplit");
			if((itr = m_parameterMap.find(L"pE")) != m_parameterMap.end()){
				m_gui->setText(IDC_EDIT_EVALPARAM,itr->second);
			}
			else
				m_gui->setText(IDC_EDIT_EVALPARAM,L"70");
		}
		else
			m_gui->setText(IDC_COMBO_EVALUATION,L"CrossValidation");
	}
	else{
		m_gui->setText(IDC_COMBO_EVALUATION,L"CrossValidation");
		if((itr = m_parameterMap.find(L"pE")) != m_parameterMap.end()){
			m_gui->setText(IDC_EDIT_EVALPARAM,itr->second);
		}
		else
			m_gui->setText(IDC_EDIT_EVALPARAM,L"10");
	}

	// Parameters
	if((itr = m_parameterMap.find(L"pTrees")) != m_parameterMap.end())
		m_gui->setText(IDC_RANDFOREST_NUMTREES,itr->second);
	else
		m_gui->setText(IDC_RANDFOREST_NUMTREES,L"100");
	if((itr = m_parameterMap.find(L"pMIPN")) != m_parameterMap.end())
		m_gui->setText(IDC_RANDFOREST_MAXINST,itr->second);
	else
		m_gui->setText(IDC_RANDFOREST_MAXINST,L"10");
	if((itr = m_parameterMap.find(L"pDepth")) != m_parameterMap.end())
		m_gui->setText(IDC_RANDFOREST_TREEDEPTH,itr->second);
	else
		m_gui->setText(IDC_RANDFOREST_TREEDEPTH,L"100");
	if((itr = m_parameterMap.find(L"pSeed")) != m_parameterMap.end())
		m_gui->setText(IDC_RANDFOREST_SEED,itr->second);
	else
		m_gui->setText(IDC_RANDFOREST_SEED,L"1");
	if((itr = m_parameterMap.find(L"pK")) != m_parameterMap.end())
		m_gui->setText(IDC_RANDFOREST_NUMFEATURES,itr->second);
	else
		m_gui->setText(IDC_RANDFOREST_NUMFEATURES,L"0");

	m_gui->setText(IDC_ALGORITHM_GPUAPI,L"CUDA");
	return true;
}