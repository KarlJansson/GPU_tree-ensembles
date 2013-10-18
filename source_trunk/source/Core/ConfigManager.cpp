#include "stdafx.h"
#include "ConfigManager.h"

namespace DataMiner{
	unsigned int ConfigManager::m_timerId = 0;
	std::map<unsigned int,clock_t> ConfigManager::m_timers = std::map<unsigned int,clock_t>();
	std::vector<unsigned int> ConfigManager::m_createTimer = std::vector<unsigned int>();
	std::map<std::string,std::string> ConfigManager::m_configValues = std::map<std::string,std::string>();
	clock_t ConfigManager::m_time = 0;
	MutexPtr ConfigManager::m_timerMutex = MutexPtr(new boost::mutex);

	ConfigManager::ConfigManager(){
	}

	void ConfigManager::initialize(){
		char* buffer;
		boost::filesystem::path path = "..\\Resources\\Config.cfg";
		if(!boost::filesystem::exists(path))
			return;
		unsigned int size = boost::filesystem::file_size(path);
		buffer = new char[size];

		std::ifstream open(path.generic_string(),std::ios_base::binary);
		open.read(buffer,size);
		open.close();

		std::string line;
		unsigned int pos = 0;
		while(pos < size){
			if(buffer[pos] == '\n'){
				unsigned int middle = line.find('=');
				m_configValues[line.substr(0,middle)] = line.substr(middle+1,line.size());
				line.clear();
			}
			else
				line += buffer[pos];
			pos++;
		}
		if(!line.empty()){
			unsigned int middle = line.find('=');
			m_configValues[line.substr(0,middle)] = line.substr(middle+1,line.size());
		}

		delete[] buffer;
	}

	void ConfigManager::updateTimers(){
		m_timerMutex->lock();
			m_time = clock();

			for(unsigned int i=0; i<m_createTimer.size(); i++){
				m_timers[m_createTimer[i]] = m_time;
			}

			m_createTimer.clear();
		m_timerMutex->unlock();
	}

	unsigned int ConfigManager::startTimer(){
		m_timerMutex->lock();
			unsigned int id = m_timerId++;
			m_createTimer.push_back(id);
		m_timerMutex->unlock();
		return id;
	}
	
	void ConfigManager::removeTimer(unsigned int timer){
		m_timerMutex->lock();
			m_timers.erase(timer);
		m_timerMutex->unlock();
	}

	void ConfigManager::resetTimer(unsigned int timer){
		m_timerMutex->lock();
			m_createTimer.push_back(timer);
		m_timerMutex->unlock();
	}

	double ConfigManager::getTime(unsigned int timer){
		m_timerMutex->lock();
			double result = double(m_time-m_timers[timer])/CLOCKS_PER_SEC;
		m_timerMutex->unlock();
		return result;
	}

	void ConfigManager::writeToFile(std::string message,std::string fileName, std::ios_base::openmode mode){
		std::ofstream open(fileName,mode);
		open << message;
		open.close();
	}

	std::string ConfigManager::getSetting(std::string setting){
		return m_configValues[setting];
	}

	void ConfigManager::setSetting(std::string settingName, std::string setting){
		m_configValues[settingName] = setting;
	}
}