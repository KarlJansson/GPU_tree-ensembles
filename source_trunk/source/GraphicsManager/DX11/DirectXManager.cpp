#include "stdafx.h"
#include "DirectXManager.h"
#include "ResourceManager.h"

namespace DataMiner{
	DirectXManager::DirectXManager(){
		m_shaderCount = 0;
		m_bufferCount = 0;
	}

	DirectXManager::~DirectXManager(){
		for(unsigned int i=0; i<m_adapters.size(); i++){
			if(m_adapters[i].m_context)
				m_adapters[i].m_context->Release();
			if(m_adapters[i].m_deviceHandle)
				m_adapters[i].m_deviceHandle->Release();
		}
		m_adapters.clear();
	}

	void DirectXManager::initialize(){
		UINT deviceFlags = 0;
		HRESULT result;
		IDXGIFactory* factory;
		IDXGIAdapter* adapter;
		IDXGIOutput* adapterOutput;
		DXGI_OUTPUT_DESC outputDesc;

#ifdef _DEBUG
		deviceFlags = D3D11_CREATE_DEVICE_DEBUG;
#endif
		// Create a DirectX graphics interface factory.
		result = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory);
		if(FAILED(result)){
			return;
		}

		bool multiCard = false;
		unsigned int	adapterId = 0,
						adapterOutputId = 0;
		while(factory->EnumAdapters(adapterId, &adapter) == S_OK){
			// Create devices
			m_adapters.push_back(dxAdapter());
			adapterOutputId = 0;
			while(adapter->EnumOutputs(adapterOutputId,&adapterOutput) == S_OK){
				if(adapterOutput->GetDesc(&outputDesc) == S_OK){
					m_adapters.back().m_name = outputDesc.DeviceName;
				}
				adapterOutput->Release();
				adapterOutputId++;
			}

			if(FAILED( result =	D3D11CreateDevice(adapter,
										D3D_DRIVER_TYPE_UNKNOWN,
										NULL,
										deviceFlags,
										NULL,
										0,
										D3D11_SDK_VERSION,
										&m_adapters.back().m_deviceHandle,
										&m_adapters.back().m_featureLevel,
										&m_adapters.back().m_context))){
				// Error handling
				//assert(0);
				return;
			}

			adapter->Release();
			adapterId++;
			if(!multiCard)
				break;
		}
		factory->Release();

		//D3D11_FEATURE_DATA_DOUBLES doubleSupport;
		//m_adapters.back().m_deviceHandle->CheckFeatureSupport(D3D11_FEATURE_DOUBLES,&doubleSupport,sizeof(doubleSupport));
	}

	void DirectXManager::launchComputation(int devId, int x, int y, int z){
		m_adapters[devId].m_context->Dispatch(1+int(float(x)/float(thread_group_size)),1+int(float(y)/float(thread_group_size)),1+int(float(z)/float(thread_group_size)));
	}

	void DirectXManager::setGPUProgram(int devId, int pid){
		m_adapters[devId].m_context->CSSetShader(m_computeShaders[pid],NULL,0);
	}

	void DirectXManager::setGPUBuffer(int devId, std::vector<int> resId, std::vector<ResourceType> rType){
		if(rType.size() != resId.size()){
			assert(0);
		}

		std::vector<ID3D11Buffer*> constantBuffer;
		std::vector<ID3D11ShaderResourceView*> resources;
		std::vector<ID3D11UnorderedAccessView*> writeBuffers;
		
		for(unsigned int i=0; i<rType.size(); i++){
			switch(rType[i]){
			case GRT_CBUFFER:
				constantBuffer.push_back(m_buffers[resId[i]].buffer);
				break;
			case GRT_RESOURCE:
				resources.push_back(m_buffers[resId[i]].rView);
				break;
			case GRT_WBUFFER:
				writeBuffers.push_back(m_buffers[resId[i]].aView);
				break;
			}
		}

		if(!constantBuffer.empty())
			setComputeShaderConstantBuffers(devId,constantBuffer);
		if(!resources.empty())
			setComputeShaderResourceViews(devId,resources);
		if(!writeBuffers.empty())
			setComputeShaderUnorderedAccessViews(devId,writeBuffers);
	}

	void DirectXManager::copyFromGPU(int devId, int resId, void* buff, unsigned int byteSize){
		D3D11_BUFFER_DESC desc;
		m_buffers[resId].buffer->GetDesc(&desc);

		desc.Usage = D3D11_USAGE_STAGING;
		desc.BindFlags = 0;
		desc.MiscFlags = 0;
		desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

		ID3D11Buffer *cpuBuffer;
		m_adapters[devId].m_deviceHandle->CreateBuffer(&desc,0,&cpuBuffer);
		m_adapters[devId].m_context->CopyResource(cpuBuffer,m_buffers[resId].buffer);

		D3D11_MAPPED_SUBRESOURCE mappedResource;
		m_adapters[devId].m_context->Map(cpuBuffer,0,D3D11_MAP_READ,0,&mappedResource);
			
		memcpy(buff,mappedResource.pData,byteSize);
			
		m_adapters[devId].m_context->Unmap(cpuBuffer,0);
		cpuBuffer->Release();
	}
	
	void DirectXManager::copyToGPU(int devId, int resId, void* buff, unsigned int byteSize){
		D3D11_MAPPED_SUBRESOURCE mappedResource;
		m_adapters[devId].m_context->Map(m_buffers[resId].buffer,0,D3D11_MAP_WRITE_DISCARD,0,&mappedResource);

		memcpy(mappedResource.pData,buff,byteSize);
			
		m_adapters[devId].m_context->Unmap(m_buffers[resId].buffer,0);
	}

	int DirectXManager::createBuffer(int devId, bufferFlags flags, UINT numElements, void* initData, unsigned int byteSize, ResourceType bType, ResourceFormat bForm, GraphicsFunctionPtr func){
		m_buffers[m_bufferCount++];
		switch(bType){
		case GRT_CBUFFER:
			createConstantBuffer(devId,&m_buffers[m_bufferCount-1].buffer,initData,byteSize);
			break;
		case GRT_RESOURCE:
			createStructuredBufferSRV(devId,flags.dxUsageFlags,flags.dxAccessFlags,numElements,&m_buffers[m_bufferCount-1].buffer,&m_buffers[m_bufferCount-1].rView,initData,byteSize);
			break;
		case GRT_WBUFFER:
			createStructuredBufferUAV(devId,flags.dxUsageFlags,flags.dxAccessFlags,numElements,&m_buffers[m_bufferCount-1].buffer,&m_buffers[m_bufferCount-1].aView,initData,byteSize);
			break;
		}
		return m_bufferCount-1;
	}
	
	int DirectXManager::createGPUProgram(int devId, std::string filename, GraphicsFunctionPtr func){
		createComputeShader(devId,filename+".hlsl",&m_computeShaders[m_shaderCount++]);
		return m_shaderCount-1;
	}

	void DirectXManager::deleteBuffer(int devId, int resId){
		if(m_buffers[resId].aView)
			m_buffers[resId].aView->Release();
		if(m_buffers[resId].rView)
			m_buffers[resId].rView->Release();
		if(m_buffers[resId].buffer)
			m_buffers[resId].buffer->Release();

		m_buffers.erase(resId);
	}
	void DirectXManager::deleteGPUProgram(int devId, int pid){
		if(m_computeShaders.find(pid) != m_computeShaders.end()){
			m_computeShaders[pid]->Release();
			m_computeShaders.erase(pid);
		}
	}

	void DirectXManager::setComputeShader(int devId, ID3D11ComputeShader* shader){
		m_adapters[devId].m_context->CSSetShader(shader,NULL,0);
	}
	
	void DirectXManager::setComputeShaderConstantBuffers(int devId, std::vector<ID3D11Buffer*> &buffers){
		m_adapters[devId].m_context->CSSetConstantBuffers(0,buffers.size(),&buffers[0]);
	}

	void DirectXManager::setComputeShaderResourceViews(int devId, std::vector<ID3D11ShaderResourceView*> &views){
		m_adapters[devId].m_context->CSSetShaderResources(0,views.size(),&views[0]);
	}
	
	void DirectXManager::setComputeShaderUnorderedAccessViews(int devId, std::vector<ID3D11UnorderedAccessView*> &views){
		unsigned int viewInt = -1;
		m_adapters[devId].m_context->CSSetUnorderedAccessViews(0,views.size(),&views[0],&viewInt);
	}

	HRESULT DirectXManager::createComputeShader(int devId, std::string filename, ID3D11ComputeShader** ppShader){
		boost::filesystem::path path = ResourceManager::findFilePath(filename);
		std::ifstream input(path.generic_string(),std::ios_base::binary);
		unsigned int size = boost::filesystem::file_size(path);

		char* buffer = new char[size];
		input.read(buffer,size);

		HRESULT hr = m_adapters[devId].m_deviceHandle->CreateComputeShader(buffer,size,NULL,ppShader);
		delete[] buffer;
		input.close();

		return hr;
	}

	HRESULT DirectXManager::createStructuredBufferUAV(int devId, DWORD usage, DWORD access, UINT iNumElements, ID3D11Buffer** ppBuffer, ID3D11UnorderedAccessView** ppUAV, const void* pInitialData, int byteSize){
		HRESULT hr = S_OK;

		// Create SB
		D3D11_BUFFER_DESC bufferDesc;
		ZeroMemory( &bufferDesc, sizeof(bufferDesc) );
		bufferDesc.ByteWidth = byteSize;
		bufferDesc.Usage = D3D11_USAGE(usage);
		bufferDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
		bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
		bufferDesc.StructureByteStride = byteSize/iNumElements;
		bufferDesc.CPUAccessFlags = access;

		D3D11_SUBRESOURCE_DATA bufferInitData;
		ZeroMemory( &bufferInitData, sizeof(bufferInitData) );
		bufferInitData.pSysMem = pInitialData;
		hr = m_adapters[devId].m_deviceHandle->CreateBuffer( &bufferDesc, (pInitialData)? &bufferInitData : NULL, ppBuffer);
		if(FAILED(hr))
			return hr;

		// Create UAV
		D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;
		ZeroMemory( &uavDesc, sizeof(uavDesc) );
		uavDesc.Format = DXGI_FORMAT_UNKNOWN;
		uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
		uavDesc.Buffer.NumElements = iNumElements;
		hr = m_adapters[devId].m_deviceHandle->CreateUnorderedAccessView( *ppBuffer, &uavDesc, ppUAV );
		if(FAILED(hr))
			return hr;

		return hr;
	}

	HRESULT DirectXManager::createStructuredBufferSRV(int devId, DWORD usage, DWORD access, UINT iNumElements, ID3D11Buffer** ppBuffer, ID3D11ShaderResourceView** ppSRV, const void* pInitialData, int byteSize){
		HRESULT hr = S_OK;

		// Create SB
		D3D11_BUFFER_DESC bufferDesc;
		ZeroMemory( &bufferDesc, sizeof(bufferDesc) );
		bufferDesc.ByteWidth = byteSize;
		bufferDesc.Usage = D3D11_USAGE(usage);
		bufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
		bufferDesc.StructureByteStride = byteSize/iNumElements;
		bufferDesc.CPUAccessFlags = access;

		D3D11_SUBRESOURCE_DATA bufferInitData;
		ZeroMemory( &bufferInitData, sizeof(bufferInitData) );
		bufferInitData.pSysMem = pInitialData;
		hr = m_adapters[devId].m_deviceHandle->CreateBuffer( &bufferDesc, (pInitialData)? &bufferInitData : NULL, ppBuffer);
		if(FAILED(hr))
			return hr;

		// Create UAV
		D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
		ZeroMemory( &srvDesc, sizeof(srvDesc) );
		srvDesc.Format = DXGI_FORMAT_UNKNOWN;
		srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
		srvDesc.Buffer.NumElements = iNumElements;
		hr = m_adapters[devId].m_deviceHandle->CreateShaderResourceView( *ppBuffer, &srvDesc, ppSRV );
		if(FAILED(hr))
			return hr;

		return hr;
	}

	HRESULT DirectXManager::createAppendConsumeBuffer(int devId, DWORD usage, DWORD access, UINT iNumElements, ID3D11Buffer** ppBuffer, ID3D11UnorderedAccessView** ppUAV, const void* pInitialData, int byteSize){
		HRESULT hr = S_OK;

		// Create SB
		D3D11_BUFFER_DESC bufferDesc;
		ZeroMemory( &bufferDesc, sizeof(bufferDesc) );
		bufferDesc.ByteWidth = byteSize;
		bufferDesc.Usage = D3D11_USAGE(usage);
		bufferDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
		bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
		bufferDesc.StructureByteStride = byteSize/iNumElements;
		bufferDesc.CPUAccessFlags = access;

		D3D11_SUBRESOURCE_DATA bufferInitData;
		ZeroMemory( &bufferInitData, sizeof(bufferInitData) );
		bufferInitData.pSysMem = pInitialData;
		hr = m_adapters[devId].m_deviceHandle->CreateBuffer( &bufferDesc, (pInitialData)? &bufferInitData : NULL, ppBuffer);
		if(FAILED(hr))
			return hr;

		// Create UAV
		D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;
		ZeroMemory( &uavDesc, sizeof(uavDesc) );
		uavDesc.Format = DXGI_FORMAT_UNKNOWN;
		uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
		uavDesc.Buffer.NumElements = iNumElements;
		uavDesc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_APPEND;
		hr = m_adapters[devId].m_deviceHandle->CreateUnorderedAccessView( *ppBuffer, &uavDesc, ppUAV );
		if(FAILED(hr))
			return hr;

		return hr;
	}

	HRESULT DirectXManager::createConstantBuffer(int devId, ID3D11Buffer** ppBuffer, const void* pInitialData, int byteSize){
		HRESULT hr = S_OK;

		// Fill in a buffer description.
		D3D11_BUFFER_DESC cbDesc;
		cbDesc.ByteWidth = 16*((byteSize+15)/16);
		cbDesc.Usage = D3D11_USAGE_DYNAMIC;
		cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		cbDesc.MiscFlags = 0;
		cbDesc.StructureByteStride = 0;

		D3D11_SUBRESOURCE_DATA bufferInitData;
		ZeroMemory( &bufferInitData, sizeof(bufferInitData) );
		bufferInitData.pSysMem = pInitialData;

		// Create the buffer.
		m_adapters[devId].m_deviceHandle->CreateBuffer( &cbDesc, (pInitialData)? &bufferInitData : NULL, ppBuffer );
		if(FAILED(hr))
			return hr;

		return hr;
	}
}