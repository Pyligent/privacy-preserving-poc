#pragma once

#include "seal/seal.h"
#include "base64.h"
#include <msclr\marshal_cppstd.h>

using namespace System;
using namespace std;
using namespace seal;
using namespace System::Collections::Generic;
using namespace msclr::interop;

namespace Microsoft
{
	namespace Research
	{
		public ref class SEALWrapper
		{
		public:
			SEALWrapper(int slotCount);
			SEALWrapper(int slotCount, bool encryptionSetup, bool keySetup);
			~SEALWrapper();

			String^ getAdded() { return gcnew String(m_addedCipherBase64->c_str());	}
			String^ getStats() { return gcnew String(m_statsCipherBase64->c_str()); }
			String^ getSummary() { return gcnew String(m_summaryCipherBase64->c_str()); }
			String^ getMlResults() { return gcnew String(m_mlCipherBase64->c_str()); }

			String^ getGaloisKeys() { return gcnew String(FromSealObject<GaloisKeys>(*m_galKeys, true).c_str()); }
			String^ getGaloisSingleStepKeys() { return gcnew String(FromSealObject<GaloisKeys>(*m_galKeysSingleStep, true).c_str()); }
			String^ getRelinKeys() { return gcnew String(FromSealObject<RelinKeys>(*m_relinKeys, true).c_str()); }

			void LoadKeys(String^ galKeys, String^ galKeysSingleStep, String^ relinKeys);
			bool AddCiphers(String^ CipherStr1, String^ CipherStr2);
			bool ComputeStatsCiphers(String^ cipherStr1, String^ cipherStr2, String^ summaryMaskCipherStr, String^ cipherGyroStr);

			// The following functions are mainly used for unit testing
			String^ Encrypt(List<double>^ values);
			List<double>^ Decrypt(String ^str);

		private:
			double m_scale;
			double m_scale_small;
			int m_slotCount;
			Evaluator *m_evaluator;
			CKKSEncoder *m_encoder;
			GaloisKeys *m_galKeys;
			GaloisKeys *m_galKeysSingleStep;
			RelinKeys *m_relinKeys;

			string *m_addedCipherBase64;
			string *m_statsCipherBase64;
			string *m_summaryCipherBase64;
            string *m_mlCipherBase64;

			template<class T>
			void ToSealObject(string input, T &output, bool base64);
			template<class T>
			string FromSealObject(T input, bool base64);

			// The following variables are mainly used for unit testing
			PublicKey *m_publicKey;
			SecretKey *m_secretKey;
			Encryptor *m_encryptor;
			Decryptor *m_decryptor;
		};
	}
}
