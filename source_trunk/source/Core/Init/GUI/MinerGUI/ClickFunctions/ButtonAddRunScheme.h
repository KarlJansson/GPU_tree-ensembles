#pragma once
#include "Runnable.h"
#include "MainFramework.h"
#include "StartAlgorithmMessage.h"

namespace DataMiner{
	class ButtonAddRunScheme : public Runnable{
	public:
		void run(void* params){
			MainFramework *mainFramework = (MainFramework*)params;
			GUIManagerPtr guiPtr = mainFramework->getGuiPtr();
			
			std::string entry;

			// Get algorithm
			entry = guiPtr->getEditText(IDC_COMBO_ALGO);
			if(entry.find("RandomForest") != std::string::npos){
				// Get data file path
				entry += " " + guiPtr->getEditText(IDC_EDIT_FILEPATH);
				// Get parameters
				entry += " " + guiPtr->getEditText(IDC_RANDFOREST_ITSELECTOR);
				entry += " " + guiPtr->getEditText(IDC_RANDFOREST_NUMTREES);
				entry += " " + guiPtr->getEditText(IDC_RANDFOREST_NUMFEATURES);
				entry += " " + guiPtr->getEditText(IDC_RANDFOREST_MAXINST);
				entry += " " + guiPtr->getEditText(IDC_RANDFOREST_TREEDEPTH);
			}
			else if(entry.find("SVM") != std::string::npos){
				
			}
			// Get evaluation form
			entry += " " + guiPtr->getEditText(IDC_COMBO_EVALUATION);
			entry += " " + guiPtr->getEditText(IDC_EDIT_EVALPARAM);

			// Add entry to scheme
			std::vector<std::wstring> items;
			items.push_back(std::wstring(entry.begin(),entry.end()));
			guiPtr->getWindow(IDC_SCHEME_COMBO_ITEMS)->addItemsToWindow(items);
		}
	private:
	};
}