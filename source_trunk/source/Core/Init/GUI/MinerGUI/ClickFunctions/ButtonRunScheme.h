#pragma once
#include "Runnable.h"
#include "MainFramework.h"
#include "StartAlgorithmMessage.h"
#include "ButtonRunAlgorithm.h"
#include "ComboAlgorithmChoice.h"

namespace DataMiner{
	class ButtonRunScheme : public Runnable{
	public:
		ButtonRunScheme():m_loadThread(ThreadPtr()){}

		void run(void* params){
			mainFramework = (MainFramework*)params;
			if(m_loadThread)
				m_loadThread->join();

			m_loadThread = ThreadPtr(new boost::thread(boost::function<void(void)>(boost::bind(&ButtonRunScheme::runScheme,this))));
		}
	private:
		void runScheme(){
			GUIManagerPtr guiPtr = mainFramework->getGuiPtr();
			RunnablePtr updateGUI = RunnablePtr(new ComboAlgorithmChoice);
			StartAlgorithmMessagePtr saPtr;
			
			unsigned int i = guiPtr->getWindow(IDC_SCHEME_COMBO_ITEMS)->getNumberOfItems();
			std::string token;

			for(unsigned int n = 0; n<i; ++n){
				guiPtr->getWindow(IDC_SCHEME_COMBO_ITEMS)->selectItemInWindow(n);
				std::stringstream stream(guiPtr->getEditText(IDC_SCHEME_COMBO_ITEMS));

				// Get algorithm
				stream >> token;
				if(token.find("RandomForest") != std::string::npos){
					guiPtr->getWindow(IDC_COMBO_ALGO)->selectItemInWindow(std::wstring(token.begin(),token.end()));
					stream >> token;
					guiPtr->setText(IDC_EDIT_FILEPATH,std::wstring(token.begin(),token.end()));
					stream >> token;
					guiPtr->getWindow(IDC_RANDFOREST_ITSELECTOR)->selectItemInWindow(std::wstring(token.begin(),token.end()));
					stream >> token;
					guiPtr->setText(IDC_RANDFOREST_NUMTREES,std::wstring(token.begin(),token.end()));
					stream >> token;
					guiPtr->setText(IDC_RANDFOREST_NUMFEATURES,std::wstring(token.begin(),token.end()));
					stream >> token;
					guiPtr->setText(IDC_RANDFOREST_MAXINST,std::wstring(token.begin(),token.end()));
					stream >> token;
					guiPtr->setText(IDC_RANDFOREST_TREEDEPTH,std::wstring(token.begin(),token.end()));
				}
				else if(token.find("SVM") != std::string::npos){
				
				}
				// Get evaluation form
				stream >> token;
				guiPtr->getWindow(IDC_COMBO_EVALUATION)->selectItemInWindow(std::wstring(token.begin(),token.end()));
				stream >> token;
				guiPtr->setText(IDC_EDIT_EVALPARAM,std::wstring(token.begin(),token.end()));

				updateGUI->run(mainFramework);

				// Run algorithm
				mainFramework->getMinerGUI()->disableAllButStop();
				AlgorithmDataPackPtr data = AlgorithmDataPackPtr(new DataMiner::AlgorithmDataPack);
				data->m_algoName = guiPtr->getEditText(IDC_COMBO_ALGO);
				std::stringstream ss;
				ss << guiPtr->getEditText(IDC_EDIT_FILEPATH);
				data->m_dataResource = ss.str();

				saPtr = StartAlgorithmMessagePtr(new StartAlgorithmMessage(data,1));
				mainFramework->postMessage(saPtr);
				saPtr->waitOnMessage();
			}
		}

		MainFramework *mainFramework;
		ThreadPtr m_loadThread;
	};
}