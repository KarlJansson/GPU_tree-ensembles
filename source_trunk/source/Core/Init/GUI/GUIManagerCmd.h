#pragma once
#include "GUIManager.h"

namespace DataMiner{
	class GUIManagerCmd : public GUIManager{
	public:
		GUIManagerCmd();

		void setText(int id, std::wstring message);
		std::string getEditText(int id);
		void postDebugMessage(std::wstring message);

		void setProgressBar(int id, unsigned int max, unsigned int progress) {}

		unsigned int addWindow(int id, std::wstring type, DWORD style1, DWORD style2, int x, int y, int width, int height, std::wstring text);

		void processWindowsMessage(int wmid, void* params){}
	protected:
		std::map<int,std::wstring> m_settings;
	};
}