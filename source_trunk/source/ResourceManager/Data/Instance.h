#pragma once
#include "Value.h"

namespace DataMiner{
	class Instance{
	public:
		Instance(Value* classValue, unsigned int size, unsigned int index);

		int classValue();
		Value::v_precision weight();
		unsigned int numValues();
		Value::v_precision getValue(unsigned int ind, bool replaceMissing = true, bool normalized = false);
		bool missing(unsigned int ind);

		unsigned int getIndex();
		DataDocumentPtr getMotherDocument() { return m_motherDocument; }
	private:
		Value* m_classValue;

		DataDocumentPtr m_motherDocument;

		unsigned int m_index;
		Value::v_precision m_weight;

		friend class DataDocument;
	};
}