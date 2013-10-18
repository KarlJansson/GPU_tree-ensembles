#pragma once
#include "Runnable.h"

namespace DataMiner{
	class GUIWindow{
	public:
		GUIWindow(int id, std::wstring type, HWND hwnd, DWORD style1, DWORD style2, int x, int y, int width, int height, HINSTANCE instance, std::wstring text);
		~GUIWindow();

		void addItemsToWindow(std::vector<std::wstring> &items);
		void removeItemsFromWindow(std::vector<std::wstring> &items);
		void selectItemInWindow(std::wstring item);
		void selectItemInWindow(int item);
		int getNumberOfItems();

		void show();
		void hide();
		void enable();
		void disable();

		void setPosition(int x, int y);
		void setDrawOrder(HWND insertOrder);

		std::pair<int,int> getPosition();
		HWND getHWND() {return m_hwnd;}

		void setOnClickFunction(RunnablePtr fn);
		void useOnClickFunction(void *params);
	private:
		HWND	m_hwnd,
				m_insertOrder;

		int m_x,
			m_y,
			m_width,
			m_height;

		RunnablePtr m_clickFunction;
	};
}

typedef boost::shared_ptr<DataMiner::GUIWindow> GUIWindowPtr;