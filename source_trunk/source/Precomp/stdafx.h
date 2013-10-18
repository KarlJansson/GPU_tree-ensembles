// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#pragma comment(linker,"\"/manifestdependency:type='win32' \
name='Microsoft.Windows.Common-Controls' version='6.0.0.0' \
processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

#include "targetver.h"

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files:
#include <windows.h>
#include "psapi.h"

// C RunTime Header Files
#include <stdlib.h>
#include <malloc.h>
#include <memory.h>
#include <tchar.h>

#include <vector>
#include <map>
#include <set>
#include <string>
#include <queue>
#include <fstream>
#include <algorithm>
#include <io.h>
#include <math.h>

#define BOOST_FILESYSTEM_NO_DEPRECATED
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/timer.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>

#include <d3d11.h>

// TODO: reference additional headers your program requires here
#include "ThreadManager.h"

#include "customTypes.h"
#include "WindowIds.h"

#include "ConfigManager.h"

#define TRACE_DEBUG(msg) std::cout << msg << std::endl;