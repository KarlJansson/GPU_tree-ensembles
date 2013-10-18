#include "stdafx.h"
#include "OpenCLManager.h"
#include "ResourceManager.h"

namespace DataMiner{
	OpenCLManager::OpenCLManager(){
		m_resourceIdCount = 0;
		m_cQueue = NULL;
		m_context = NULL;
	}

	OpenCLManager::~OpenCLManager(){
		if(m_cQueue)
			clReleaseCommandQueue(m_cQueue);
		if(m_context)
			clReleaseContext(m_context);
		clUnloadCompiler();
	}

	void OpenCLManager::initialize(){
		// Get OpenCL platform count
		cl_uint NumPlatforms;
		clGetPlatformIDs(0, NULL, &NumPlatforms);

		// Get all OpenCL platform IDs
		cl_platform_id* PlatformIDs;
		PlatformIDs = new cl_platform_id[NumPlatforms];
		clGetPlatformIDs(NumPlatforms, PlatformIDs, NULL);

		// Select NVIDIA platform
		bool foundPlatform = false;
		char cBuffer[1024];
		cl_uint NvPlatform;
		for(cl_uint i = 0; i < NumPlatforms; ++i){
			clGetPlatformInfo(PlatformIDs[i], CL_PLATFORM_NAME, 1024, cBuffer, NULL);
			if(strstr(cBuffer, "NVIDIA") != NULL){
				NvPlatform = i;
				foundPlatform = true;
				break;
			}
		}

		if(!foundPlatform)
			return;

		// Get a GPU device on Platform
		cl_device_id cdDevice;
		clGetDeviceIDs(PlatformIDs[NvPlatform], CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
		delete[] PlatformIDs;

		cl_int errorCode;
		// Create a context
		m_context = clCreateContext(0, 1, &cdDevice, NULL, NULL, &errorCode);
		assert(errorCode == CL_SUCCESS);

		// Create a command queue for the device in the context
		m_cQueue = clCreateCommandQueue(m_context, cdDevice, 0, &errorCode);
		assert(errorCode == CL_SUCCESS);
	}

	void OpenCLManager::launchComputation(int devId, int x, int y, int z){
		size_t threadsInGroup = thread_group_size;
		size_t totalThreads = (((x*y*z)/threadsInGroup)+1)*threadsInGroup;
		cl_int errorCode = clEnqueueNDRangeKernel(m_cQueue, m_kernels[m_currentKernel], 1, 0, &totalThreads, &threadsInGroup, 0, 0, 0);
		assert(errorCode == CL_SUCCESS);
	}

	void OpenCLManager::setGPUProgram(int devId, int pid){
		m_currentKernel = pid;
	}
	
	void OpenCLManager::setGPUBuffer(int devId, std::vector<int> resId, std::vector<ResourceType> rType){
		cl_int errorCode;
		for(unsigned int i=0; i<resId.size(); i++){
			errorCode = clSetKernelArg(m_kernels[m_currentKernel], i, sizeof(cl_mem), (void *)&m_buffers[resId[i]]);
			assert(errorCode == CL_SUCCESS);
		}
	}

	int OpenCLManager::getNumDevices(){
		return 1;
	}

	void OpenCLManager::copyFromGPU(int devId, int resId, void* buff, unsigned int byteSize){
		cl_int errorCode;
		errorCode = clEnqueueReadBuffer(m_cQueue, m_buffers[resId], CL_TRUE, 0, byteSize, buff, 0, 0, 0);
		assert(errorCode == CL_SUCCESS);
	}

	void OpenCLManager::copyToGPU(int devId, int resId, void* buff, unsigned int byteSize){
		cl_int errorCode;
		errorCode = clEnqueueWriteBuffer(m_cQueue,m_buffers[resId],CL_TRUE,0,byteSize,buff,0,NULL,NULL);
		assert(errorCode == CL_SUCCESS);
	}

	int OpenCLManager::createBuffer(int devId, bufferFlags flags, UINT numElements, void* initData, unsigned int byteSize, ResourceType bType, ResourceFormat bForm, GraphicsFunctionPtr func){
		cl_int errorCode;
		int byteAligned = byteSize;
		if(bType == GRT_CBUFFER)
			byteAligned = 16*((byteSize+15)/16);
		m_buffers[m_resourceIdCount++] = clCreateBuffer(m_context, flags.openCLFlags, byteAligned, initData, &errorCode);
		assert(errorCode == CL_SUCCESS);

		if(initData)
			copyToGPU(devId,m_resourceIdCount-1,initData,byteSize);

		return m_resourceIdCount-1;
	}

	int OpenCLManager::createGPUProgram(int devId, std::string filename, GraphicsFunctionPtr func){
		boost::filesystem::path path = ResourceManager::findFilePath(filename+".cl");
		std::ifstream input(path.generic_string(),std::ios_base::binary);
		size_t size = boost::filesystem::file_size(path);

		char* buffer = new char[size];
		input.read(buffer,size);

		cl_int errorCode;
		m_programs[m_resourceIdCount++] = clCreateProgramWithSource(m_context, 1, (const char**)&buffer, (const size_t*)&size, &errorCode);
		assert(errorCode == CL_SUCCESS);

		input.close();
		delete[] buffer;

		errorCode = clBuildProgram(m_programs[m_resourceIdCount-1], 0, 0, "-I ../Resources/Shaders", 0, 0);
		assert(errorCode == CL_SUCCESS);

		// Create kernel instance
		m_kernels[m_resourceIdCount-1] = clCreateKernel(m_programs[m_resourceIdCount-1], "cl_entry", &errorCode);
		assert(errorCode == CL_SUCCESS);

		return m_resourceIdCount-1;
	}

	void OpenCLManager::deleteBuffer(int devId, int resId){
		cl_int errorCode;
		errorCode = clReleaseMemObject(m_buffers[resId]);
		assert(errorCode == CL_SUCCESS);
		m_buffers.erase(resId);
	}
	
	void OpenCLManager::deleteGPUProgram(int devId, int pid){
		cl_int errorCode;
		errorCode = clReleaseKernel(m_kernels[pid]);
		assert(errorCode == CL_SUCCESS);
		m_kernels.erase(pid);
		errorCode = clReleaseProgram(m_programs[pid]);
		assert(errorCode == CL_SUCCESS);
		m_programs.erase(pid);
	}

	void OpenCLManager::checkError(cl_int error){
		std::string errorDesc = "";
		switch(error){
		case CL_INVALID_PROGRAM_EXECUTABLE:
			errorDesc = "CL_INVALID_PROGRAM_EXECUTABLE";
			break;
		case CL_INVALID_COMMAND_QUEUE:
			errorDesc = "CL_INVALID_COMMAND_QUEUE";
			break;
		case CL_INVALID_KERNEL:
			errorDesc = "CL_INVALID_KERNEL";
			break;
		case CL_INVALID_CONTEXT:
			errorDesc = "CL_INVALID_CONTEXT";
			break;
		case CL_INVALID_KERNEL_ARGS:
			errorDesc = "CL_INVALID_KERNEL_ARGS";
			break;
		case CL_INVALID_WORK_DIMENSION:
			errorDesc = "CL_INVALID_WORK_DIMENSION";
			break;
		case CL_INVALID_GLOBAL_WORK_SIZE:
			errorDesc = "CL_INVALID_GLOBAL_WORK_SIZE";
			break;
		case CL_INVALID_GLOBAL_OFFSET:
			errorDesc = "CL_INVALID_GLOBAL_OFFSET";
			break;
		case CL_INVALID_WORK_GROUP_SIZE:
			errorDesc = "CL_INVALID_WORK_GROUP_SIZE";
			break;
		case CL_INVALID_WORK_ITEM_SIZE:
			errorDesc = "CL_INVALID_WORK_ITEM_SIZE";
			break;
		case CL_MISALIGNED_SUB_BUFFER_OFFSET:
			errorDesc = "CL_MISALIGNED_SUB_BUFFER_OFFSET";
			break;
		case CL_INVALID_IMAGE_SIZE:
			errorDesc = "CL_INVALID_IMAGE_SIZE";
			break;
		case CL_OUT_OF_RESOURCES:
			errorDesc = "CL_OUT_OF_RESOURCES";
			break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			errorDesc = "CL_MEM_OBJECT_ALLOCATION_FAILURE";
			break;
		case CL_INVALID_EVENT_WAIT_LIST:
			errorDesc = "CL_INVALID_EVENT_WAIT_LIST";
			break;
		case CL_OUT_OF_HOST_MEMORY:
			errorDesc = "CL_OUT_OF_HOST_MEMORY";
			break;
		default:
			errorDesc = "UNKNOWN";
			break;
		}
	}
}