#pragma once
#include "GraphicsManager.h"

namespace DataMiner{
	class DirectXManager : public GraphicsManager{
	public:
		DirectXManager();
		~DirectXManager();

		struct dxAdapter{
			ID3D11Device			*m_deviceHandle;
			ID3D11DeviceContext		*m_context;
			D3D_FEATURE_LEVEL		m_featureLevel;

			std::wstring m_name;
		};

		void initialize();

		int createDeviceContext(int devId){return 0;}
		void destroyDeviceContext(int id){}

		void launchComputation(int devId, int x, int y, int z);
		void launchComputation(int x, int y, int z, bool split = true){launchComputation(0,x,y,z);}
		void syncDevice(int devId){}
		void syncDevice(){syncDevice(0);}

		void setGPUProgram(int devId, int pid);
		void setGPUProgram(int pid){setGPUProgram(0,pid);}
		void setGPUBuffer(int devId, std::vector<int> resId, std::vector<ResourceType> rType);
		void setGPUBuffer(std::vector<int> resId, std::vector<ResourceType> rType) {setGPUBuffer(0,resId,rType);}
		
		int getNumDevices() { return 1; }

		void copyFromGPU(int devId, int resId, void* buff, unsigned int byteSize);
		void copyFromGPU(int resId, void* buff, unsigned int byteSize, unsigned int elements, bool split = true){copyFromGPU(0,resId,buff,byteSize);}
		void copyToGPU(int devId, int resId, void* buff, unsigned int byteSize);
		void copyToGPU(int resId, void* buff, unsigned int byteSize, unsigned int elements, bool split = true){copyToGPU(0,resId,buff,byteSize);}

		int createBuffer(int devId, bufferFlags flags, UINT numElements, void* initData, unsigned int byteSize, ResourceType bType, ResourceFormat bForm = GRF_REAL, GraphicsFunctionPtr func = GraphicsFunctionPtr());
		int createBuffer(bufferFlags flags, UINT numElements, void* initData, unsigned int byteSize, ResourceType bType, ResourceFormat bForm = GRF_REAL, GraphicsFunctionPtr func = GraphicsFunctionPtr(), bool split = true){return createBuffer(0,flags,numElements,initData,byteSize,bType,bForm,func);}
		int createGPUProgram(int devId, std::string filename, GraphicsFunctionPtr func = GraphicsFunctionPtr());
		int createGPUProgram(std::string filename, GraphicsFunctionPtr func = GraphicsFunctionPtr()){return createGPUProgram(0,filename,func);}
		void deleteBuffer(int devId, int resId);
		void deleteBuffer(int resId){deleteBuffer(0,resId);}
		void deleteGPUProgram(int devId, int pid);
		void deleteGPUProgram(int pid){deleteGPUProgram(0,pid);}

		int markGPUTime(int devId){return 0;}
		int markGPUTime(){return markGPUTime(0);}
		float getGPUTime(int devId, int timer){return 0;}
		float getGPUTime(int timer){return getGPUTime(0,timer);}

	private:
		void createShaders();

		HRESULT createStructuredBufferUAV(int devId, DWORD usage, DWORD access, UINT iNumElements, ID3D11Buffer** ppBuffer, ID3D11UnorderedAccessView** ppUAV, const void* pInitialData, int byteSize);
		HRESULT createStructuredBufferSRV(int devId, DWORD usage, DWORD access, UINT iNumElements, ID3D11Buffer** ppBuffer, ID3D11ShaderResourceView** ppSRV, const void* pInitialData, int byteSize);
		HRESULT createAppendConsumeBuffer(int devId, DWORD usage, DWORD access, UINT iNumElements, ID3D11Buffer** ppBuffer, ID3D11UnorderedAccessView** ppUAV, const void* pInitialData, int byteSize);
		HRESULT createConstantBuffer(int devId, ID3D11Buffer** ppBuffer, const void* pInitialData, int byteSize);

		std::vector<dxAdapter>	m_adapters;

		struct bufferStruct{
			bufferStruct():buffer(NULL),rView(NULL),aView(NULL){}
			~bufferStruct(){
				if(buffer)
					buffer->Release();
				if(rView)
					rView->Release();
				if(aView)
					aView->Release();

				buffer = NULL;
				rView = NULL;
				aView = NULL;
			}

			ID3D11Buffer* buffer;
			ID3D11ShaderResourceView* rView;
			ID3D11UnorderedAccessView* aView;
		};

		std::map<int,ID3D11ComputeShader*> m_computeShaders;
		std::map<int,bufferStruct> m_buffers;

		int m_shaderCount;
		int m_bufferCount;

		HWND m_hwnd;

		void setComputeShader(int devId, ID3D11ComputeShader* shader);
		void setComputeShaderConstantBuffers(int devId, std::vector<ID3D11Buffer*> &buffers);
		void setComputeShaderResourceViews(int devId, std::vector<ID3D11ShaderResourceView*> &views);
		void setComputeShaderUnorderedAccessViews(int devId, std::vector<ID3D11UnorderedAccessView*> &views);

		HRESULT createComputeShader(int devId, std::string filename, ID3D11ComputeShader** ppShader);
	};
}

typedef boost::shared_ptr<DataMiner::DirectXManager> DirectXManagerPtr;