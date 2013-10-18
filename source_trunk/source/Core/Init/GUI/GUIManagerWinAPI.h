#pragma once
#include "GUIManager.h"

namespace DataMiner{
	class GUIManagerWinAPI : public GUIManager{
	public:
		GUIManagerWinAPI(HWND hwnd, HINSTANCE hInst);

		void setText(int id, std::wstring message);
		std::string getEditText(int id);
		void setProgressBar(int id, unsigned int max, unsigned int progress);

		void postDebugMessage(std::wstring message);

		unsigned int addWindow(int id, std::wstring type, DWORD style1, DWORD style2, int x, int y, int width, int height, std::wstring text);

		void processWindowsMessage(int wmid, void* params);
	private:
		HWND m_hwnd;
		HINSTANCE m_hinstance;
	};
}