#pragma once
#include "Runnable.h"
#include "MainFramework.h"
#include "StartAlgorithmMessage.h"

namespace DataMiner{
	class ButtonRunAlgorithm : public Runnable{
	public:
		void run(void* params){
			MainFramework *mainFramework = (MainFramework*)params;
			GUIManagerPtr guiPtr = mainFramework->getGuiPtr();
			mainFramework->getMinerGUI()->disableAllButStop();

			AlgorithmDataPackPtr data = AlgorithmDataPackPtr(new DataMiner::AlgorithmDataPack);
			data->m_algoName = guiPtr->getEditText(IDC_COMBO_ALGO);
			std::stringstream ss;
			ss << guiPtr->getEditText(IDC_EDIT_FILEPATH);
			data->m_dataResource = ss.str();

			mainFramework->postMessage(StartAlgorithmMessagePtr(new StartAlgorithmMessage(data)));
		}
	private:
	};
}