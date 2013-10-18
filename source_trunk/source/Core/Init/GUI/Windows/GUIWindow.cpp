#include "stdafx.h"
#include "GUIWindow.h"

namespace DataMiner{
	GUIWindow::GUIWindow(int id, std::wstring type, HWND hwnd, DWORD style1, DWORD style2, int x, int y, int width, int height, HINSTANCE instance, std::wstring text):m_hwnd(0){
		if(!hwnd)
			return;
		
		m_hwnd = CreateWindowEx(	style1,	
									type.c_str(),
									text.c_str(),
									style2,
									x,
									y,
									width,
									height,
									hwnd,
									(HMENU)id,
									instance,
									NULL);
		HFONT defaultFont;
		defaultFont = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
		SendMessage(m_hwnd, WM_SETFONT, WPARAM (defaultFont), TRUE);

		m_insertOrder = NULL;
		m_clickFunction = RunnablePtr();

		m_x = x;
		m_y = y;
		m_width = width;
		m_height = height;
	}

	GUIWindow::~GUIWindow(){
		if(m_hwnd)
			DestroyWindow(m_hwnd);
	}

	void GUIWindow::addItemsToWindow(std::vector<std::wstring> &items){
		if(!m_hwnd)
			return;

		for(unsigned int i=0; i<items.size(); i++)
			SendMessage(m_hwnd, CB_ADDSTRING, 0, (LPARAM)items[i].c_str());
		SendMessage(m_hwnd, CB_SETCURSEL, 0, 0);
	}

	void GUIWindow::removeItemsFromWindow(std::vector<std::wstring> &items){
		if(!m_hwnd)
			return;

		for(unsigned int i=0; i<items.size(); i++)
			SendMessage(m_hwnd, CB_DELETESTRING, 0, (LPARAM)items[i].c_str());
		SendMessage(m_hwnd, CB_SETCURSEL, 0, 0);
	}

	void GUIWindow::selectItemInWindow(int item){
		if(!m_hwnd)
			return;
		SendMessage(m_hwnd, CB_SETCURSEL, item, 0);
	}

	void GUIWindow::selectItemInWindow(std::wstring item){
		if(!m_hwnd)
			return;
		SendMessage(m_hwnd, CB_SELECTSTRING, 0, (LPARAM)item.c_str());
	}

	int GUIWindow::getNumberOfItems(){
		if(!m_hwnd)
			return 0;
		return SendMessage(m_hwnd, CB_GETCOUNT, 0, 0);
	}

	void GUIWindow::show(){
		if(!m_hwnd)
			return;
		ShowWindow(m_hwnd,SW_SHOW);
	}
	
	void GUIWindow::hide(){
		if(!m_hwnd)
			return;
		ShowWindow(m_hwnd,SW_HIDE);
	}

	void GUIWindow::enable(){
		if(!m_hwnd)
			return;
		EnableWindow(m_hwnd,true);
	}
	
	void GUIWindow::disable(){
		if(!m_hwnd)
			return;
		EnableWindow(m_hwnd,false);
	}

	void GUIWindow::setPosition(int x, int y){
		if(!m_hwnd)
			return;
		BOOL success = SetWindowPos(m_hwnd,m_insertOrder,x,y,m_width,m_height,0);
	}

	void GUIWindow::setDrawOrder(HWND insertOrder){
		if(!m_hwnd)
			return;
		m_insertOrder = insertOrder;
		BOOL success = SetWindowPos(m_hwnd,insertOrder,m_x,m_y,m_width,m_height,0);
	}

	std::pair<int,int> GUIWindow::getPosition(){
		return std::pair<int,int>(m_x,m_y);
	}

	void GUIWindow::setOnClickFunction(RunnablePtr fn){
		if(!m_hwnd)
			return;
		m_clickFunction = fn;
	}

	void GUIWindow::useOnClickFunction(void *params){
		if(!m_hwnd)
			return;
		if(m_clickFunction)
			m_clickFunction->run(params);
	}
}