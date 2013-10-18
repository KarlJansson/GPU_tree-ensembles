#pragma once
#include "Runnable.h"
#include "MainFramework.h"
#include "StopAlgorithmMessage.h"

namespace DataMiner{
	class ComboAlgorithmChoice : public Runnable{
	public:
		void run(void* params){
			MainFramework *mainFramework = (MainFramework*)params;
			std::string value = mainFramework->getGuiPtr()->getEditText(IDC_COMBO_ALGO);
			if(value.find("RandomForest") != std::string::npos){
				mainFramework->getGuiPtr()->getGroup(IDC_GROUP_RANDFOREST)->show();
				mainFramework->getGuiPtr()->getGroup(IDC_GROUP_SVM)->hide();
			}
			else{
				mainFramework->getGuiPtr()->getGroup(IDC_GROUP_RANDFOREST)->hide();
				mainFramework->getGuiPtr()->getGroup(IDC_GROUP_SVM)->show();
			}
		}
	private:
	};
}