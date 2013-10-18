#pragma once
#include "Runnable.h"
#include "MainFramework.h"
#include "StartAlgorithmMessage.h"

namespace DataMiner{
	class ButtonLoadScheme : public Runnable{
	public:
		void run(void* params){
			MainFramework *mainFramework = (MainFramework*)params;
			GUIManagerPtr guiPtr = mainFramework->getGuiPtr();

			std::string path = guiPtr->getEditText(IDC_EDIT_FILEPATH);
			std::string entry;
			std::vector<std::wstring> items;
			char buffer[512];

			std::ifstream open(path);
			if(!open.fail()){
				while(!open.eof()){
					open.getline(buffer,512);
					entry = buffer;
					items.push_back(std::wstring(entry.begin(), entry.end()));
				}
				open.close();
			}

			guiPtr->getWindow(IDC_SCHEME_COMBO_ITEMS)->addItemsToWindow(items);
		}
	private:
	};
}