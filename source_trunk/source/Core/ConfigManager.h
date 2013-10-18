#pragma once

namespace DataMiner{
	class ConfigManager{
	public:
		static void initialize();
		static unsigned int startTimer();
		static void removeTimer(unsigned int timer);
		static void resetTimer(unsigned int timer);
		static double getTime(unsigned int timer);
		static void updateTimers();
		static void writeToFile(std::string message,std::string fileName, std::ios_base::openmode mode = std::ios_base::app);

		static std::string getSetting(std::string setting);

		static void setSetting(std::string settingName, std::string setting);
	private:
		ConfigManager();

		static MutexPtr m_timerMutex;
		static std::map<unsigned int,clock_t> m_timers;
		static std::vector<unsigned int> m_createTimer;
		static clock_t m_time;

		static std::map<std::string,std::string> m_configValues;
		static unsigned int m_timerId;
	};
}