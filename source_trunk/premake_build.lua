local cudaPath = os.getenv("CUDA_PATH");
-- A solution contains projects, and defines the available configurations
solution "GPU_DataMining"
	configurations { "Debug", "Release", "Distribution" }
	location "ProjectFiles"
	
	-- A project defines one build target
	project "GPU_DataMining_CMD"
		targetname "dataMiner_cmd"
		platforms { "x32", "x64" }
		kind "ConsoleApp"
		language "C++"
		location "ProjectFiles"
		pchheader("stdafx.h")
		pchsource("./Source/Precomp/stdafx.cpp")
		
		files{ 
			"Main_Cmd/**.h",
			"Main_Cmd/**.cpp",
			"Source/**.h", 
			"Source/**.cpp",
			"kernel_code/**.cu"
		}
		
		includedirs{
			-- Internal Includes
			"./Main_Cmd",
			"./Source/Core",
			"./Source/Core/Init",
			"./Source/Core/Init/GUI",
			"./Source/Core/Init/GUI/MinerGUI",
			"./Source/Core/Init/GUI/MinerGUI/ClickFunctions",
			"./Source/Core/Init/GUI/Windows",
			"./Source/Core/Init/MiningUtilities",
			"./Source/Core/MessageManager",
			"./Source/Core/MessageManager/DataPacks",
			"./Source/Core/MessageManager/Handlers",
			"./Source/Core/MessageManager/Messages",
			"./Source/Core/ThreadManager",
			
			"./Source/Algorithms",
			"./Source/Algorithms/Evaluation",
			"./Source/Algorithms/RandomForest",
			"./Source/Algorithms/RandomForest/GPURandomForest",
			"./Source/Algorithms/RandomForest/GPURandomForest/CUDAFunctions",
			"./Source/Algorithms/RandomForest/GPURandomForest/CUDAFunctions/ConstantUpdates",
			"./Source/Algorithms/RandomForest/RecursiveCPU",
			"./Source/Algorithms/RandomForest/SerialCPU",
			"./Source/Algorithms/SVM",
			"./Source/Algorithms/SVM/Kernels",
			"./Source/Algorithms/SVM/CUDAFunctions",
			
			"./Source/GraphicsManager",
			"./Source/GraphicsManager/CUDA",
			"./Source/GraphicsManager/DX11",
			"./Source/GraphicsManager/OpenCL",
			
			"./Source/Precomp",
			
			"./Source/ResourceManager",
			"./Source/ResourceManager/Data",
			"./Source/ResourceManager/Parsers",
			"./Source/ResourceManager/Parsers/Interface",
			
			-- External includes (Change to relevant directories)
			"../../Dependencies/boost", -- Boost
			(cudaPath .. "include"), -- CUDA Toolkit
			"C:/Program Files (x86)/Microsoft SDKs/Windows/v7.0A/Include" -- Windows SDK
		}
 
		configuration {"Debug", "x64"}
			libdirs{
				"../../Dependencies/boost/stage64/lib", -- Boost x64
				(cudaPath .. "lib/x64"), -- CUDA Toolkit x64
				"C:/Program Files (x86)/Microsoft SDKs/Windows/v7.0A/Lib/x64" -- Windows SDK x64
			}
			buildoptions { "-Zm112" }
			targetdir "Build_Output/Debug"
			defines { "_DEBUG" }
			flags { "Symbols", "Unicode", "WinMain" }
			links{ "cuda", "cudart", "d3d11", "dxgi", "OpenCL", "Psapi" }
			
		configuration {"Debug", "x32"}
			libdirs{
				"../../Dependencies/boost/stage/lib", -- Boost x32
				(cudaPath .. "lib/Win32"), -- CUDA Toolkit x32
				"C:/Program Files (x86)/Microsoft SDKs/Windows/v7.0A/Lib" -- Windows SDK x32
			}
			buildoptions { "-Zm112" }
			targetdir "Build_Output/Debug"
			defines { "_DEBUG" }
			flags { "Symbols", "Unicode", "WinMain" }
			links{ "cuda", "cudart", "d3d11", "dxgi", "OpenCL", "Psapi" }
 
		configuration {"Release", "x64"}
			libdirs{
				"../../Dependencies/boost/stage64/lib", -- Boost x64
				(cudaPath .. "lib/x64"), -- CUDA Toolkit x64
				"C:/Program Files (x86)/Microsoft SDKs/Windows/v7.0A/Lib/x64" -- Windows SDK x64
			}
			buildoptions { "/MP" }
			targetdir "Build_Output/Release"
			defines { "NDEBUG" }
			flags { "Symbols", "Optimize", "Unicode", "WinMain" }
			links{ "cuda", "cudart", "cuda", "d3d11", "dxgi", "OpenCL", "Psapi" }
			
		configuration {"Release", "x32"}
			libdirs{
				"../../Dependencies/boost/stage/lib", -- Boost x32
				(cudaPath .. "lib/Win32"), -- CUDA Toolkit x32
				"C:/Program Files (x86)/Microsoft SDKs/Windows/v7.0A/Lib" -- Windows SDK x32
			}
			buildoptions { "/MP" }
			targetdir "Build_Output/Release"
			defines { "NDEBUG" }
			flags { "Symbols", "Optimize", "Unicode", "WinMain" }
			links{ "cuda", "cudart", "d3d11", "dxgi", "OpenCL", "Psapi" }
			
		configuration {"Distribution", "x64"}
			libdirs{
				"../../Dependencies/boost/stage64/lib", -- Boost x64
				(cudaPath .. "lib/x64"), -- CUDA Toolkit x64
				"C:/Program Files (x86)/Microsoft SDKs/Windows/v7.0A/Lib/x64" -- Windows SDK x64
			}
			buildoptions { "/MP" }
			targetdir "Build_Output/Distribution/x64"
			defines { "NDEBUG" }
			flags { "Optimize", "Unicode", "WinMain" }
			links{ "cuda", "cudart", "d3d11", "dxgi", "OpenCL", "Psapi" }
			
		configuration {"Distribution", "x32"}
			libdirs{
				"../../Dependencies/boost/stage/lib", -- Boost x32
				(cudaPath .. "lib/Win32"), -- CUDA Toolkit x32
				"C:/Program Files (x86)/Microsoft SDKs/Windows/v7.0A/Lib" -- Windows SDK x32
			}
			buildoptions { "/MP" }
			targetdir "Build_Output/Distribution/x32"
			defines { "NDEBUG" }
			flags { "Optimize", "Unicode", "WinMain" }
			links{ "cuda", "cudart", "d3d11", "dxgi", "OpenCL", "Psapi" }