#pragma once

namespace DataMiner{
	class Value{
	public:
		typedef float v_precision;
		#define ValueMax FLT_MAX

		Value(){}
		Value(v_precision value){
			m_value = value;
		}

		v_precision getValue() {return m_value;}
		void setValue(v_precision val) { m_value = val; }
	private:
		v_precision m_value;
	};
}