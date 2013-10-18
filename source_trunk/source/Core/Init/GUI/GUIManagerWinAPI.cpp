#include "stdafx.h"
#include "GUIManagerWinAPI.h"
#include "MainFramework.h"
#include "DirectXManager.h"
#include <CommCtrl.h>

namespace DataMiner{
	GUIManagerWinAPI::GUIManagerWinAPI(HWND hwnd, HINSTANCE hInst){
		m_hwnd = hwnd;
		m_hinstance = hInst;
	}

	void GUIManagerWinAPI::setText(int id, std::wstring message){
		HWND hwnd = m_windows[id]->getHWND();
		SetWindowText(hwnd,message.c_str());
	}

	void GUIManagerWinAPI::setProgressBar(int id, unsigned int max, unsigned int progress){
		HWND hwnd = m_windows[id]->getHWND();
		SendMessage(hwnd, PBM_SETRANGE, 0, MAKELPARAM(0, max));
		SendMessage(hwnd, PBM_SETPOS, (WPARAM)progress, 0);
	}

	std::string GUIManagerWinAPI::getEditText(int id){
		std::string result;
		HWND hwnd = m_windows[id]->getHWND();
		CHAR buff[1024];
		GetWindowTextA(hwnd, buff, 1024);
		result = buff;
		return result;
	}

	void GUIManagerWinAPI::postDebugMessage(std::wstring message){
		m_debugString += message;
		HWND hwnd = m_windows[IDC_STATIC_DEBUG]->getHWND();
		SetWindowText(hwnd,m_debugString.c_str());
	}

	unsigned int GUIManagerWinAPI::addWindow(int id, std::wstring type, DWORD style1, DWORD style2, int x, int y, int width, int height, std::wstring text){
		m_windows[id] = GUIWindowPtr(new GUIWindow(id,type,m_hwnd,style1,style2,x,y,width,height,m_hinstance,text));
		return id;
	}

	void GUIManagerWinAPI::processWindowsMessage(int wmid, void* params){
		std::map<unsigned int, GUIWindowPtr>::iterator itr = m_windows.find(wmid);
		if(itr != m_windows.end()){
			itr->second->useOnClickFunction(params);
		}
	}
}