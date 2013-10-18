#pragma once
#include "Runnable.h"
#include "MainFramework.h"
#include "StartAlgorithmMessage.h"

namespace DataMiner{
	class ButtonRemoveScheme : public Runnable{
	public:
		void run(void* params){
			MainFramework *mainFramework = (MainFramework*)params;
			GUIManagerPtr guiPtr = mainFramework->getGuiPtr();

			std::string entry = guiPtr->getEditText(IDC_SCHEME_COMBO_ITEMS);

			std::vector<std::wstring> items;
			items.push_back(std::wstring(entry.begin(),entry.end()));
			guiPtr->getWindow(IDC_SCHEME_COMBO_ITEMS)->removeItemsFromWindow(items);
		}
	private:
	};
}