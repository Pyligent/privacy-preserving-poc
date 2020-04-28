#include "stdafx.h"

#include "SEALWrapper.h"
#include <fstream>
#include <vector>

namespace Microsoft
{
    namespace Research
    {
        namespace
        {
            // Generate half the number of Galois keys
            vector<uint64_t> half_size_gal_keys(shared_ptr<SEALContext> context)
            {
                size_t coeff_count = context->context_data()->parms().poly_modulus_degree();
                uint64_t m = coeff_count << 1;
                int logn = util::get_power_of_two(static_cast<uint64_t>(coeff_count));

                vector<uint64_t> logn_galois_keys{};

                // Generate Galois key for power of 3 mod m (X -> X^{3^k})
                uint64_t two_power_of_three = 3;
                for (int i = 0; i < logn - 1; i++)
                {
                    logn_galois_keys.push_back(two_power_of_three);
                    two_power_of_three *= two_power_of_three;
                    two_power_of_three &= (m - 1);
                }
                return logn_galois_keys;
            }
        }

        SEALWrapper::SEALWrapper(int slotCount)
        {
            m_slotCount = slotCount;
            int polyModulus = 2 * m_slotCount;

            // The coeff_modulus parameter below is secure up to polyModulus == 8192.
            if (polyModulus > 8192)
            {
                throw invalid_argument("insecure parameters");
            }

            // Create the seal environment
            EncryptionParameters parms(scheme_type::CKKS);
            parms.set_poly_modulus_degree(polyModulus);
            parms.set_coeff_modulus({
                0xffffffffffc0001, // 60 bits
                0x7fffffffffcc001, // 59 bits
                0x7fffffffffa4001, // 59 bits
                0xffffe80001       // 40 bits
            });

            auto context = SEALContext::Create(parms);
            m_evaluator = new Evaluator(context);
            m_encoder = new CKKSEncoder(context);
            EncryptionParameters data_parms = context->context_data()->parms();
            m_scale = static_cast<double>(data_parms.coeff_modulus().back().value());
            m_scale_small = pow(2.0, 25);
        }

        SEALWrapper::SEALWrapper(int slotCount, bool encryptionSetup, bool keySetup)
        {
            m_slotCount = slotCount;
            int polyModulus = 2 * m_slotCount;

            // The coeff_modulus parameter below is secure up to polyModulus == 8192.
            if (polyModulus > 8192)
            {
                throw invalid_argument("insecure parameters");
            }

            // Create the seal environment
            EncryptionParameters parms(scheme_type::CKKS);
            parms.set_poly_modulus_degree(polyModulus);
            parms.set_coeff_modulus({
                0xffffffffffc0001, // 60 bits
                0x7fffffffffcc001, // 59 bits
                0x7fffffffffa4001, // 59 bits
                0xffffe80001       // 40 bits
            });

            auto context = SEALContext::Create(parms);
            m_evaluator = new Evaluator(context);
            m_encoder = new CKKSEncoder(context);
            EncryptionParameters data_parms = context->context_data()->parms();
            m_scale = static_cast<double>(data_parms.coeff_modulus().back().value());
            m_scale_small = pow(2.0, 25);

            if (encryptionSetup)
            {
                KeyGenerator keygen(context);
                m_publicKey = new PublicKey(keygen.public_key());
                m_secretKey = new SecretKey(keygen.secret_key());
                m_encryptor = new Encryptor(context, keygen.public_key());
                m_decryptor = new Decryptor(context, keygen.secret_key());
                
                if (keySetup)
                {
                    // Create rotation key for single step rotation with small dbc
                    vector<uint64_t> galoisElts = { uint64_t(3) };
                    m_galKeysSingleStep = new GaloisKeys(keygen.galois_keys(15, galoisElts));

                    // Use bigger dbc for generic Galois keys
                    m_galKeys = new GaloisKeys(keygen.galois_keys(dbc_max(),
                        half_size_gal_keys(context)));

                    m_relinKeys = new RelinKeys(keygen.relin_keys(60));
                }
            }
        }

        SEALWrapper::~SEALWrapper()
        {
            // Delete the pointers if they exist
            if (m_evaluator) { delete m_evaluator; }
            if (m_encoder) { delete m_encoder; }
            if (m_encryptor) { delete m_encryptor; }
            if (m_decryptor) { delete m_decryptor; }

            if (m_publicKey) { delete m_publicKey; }
            if (m_secretKey) { delete m_secretKey; }
            if (m_galKeys) { delete m_galKeys; }
            if (m_galKeysSingleStep) { delete m_galKeysSingleStep; }
            if (m_relinKeys) { delete m_relinKeys; }

            if (m_addedCipherBase64) { delete m_addedCipherBase64; }
            if (m_statsCipherBase64) { delete m_statsCipherBase64; }
            if (m_summaryCipherBase64) { delete m_summaryCipherBase64; }
        }

        void SEALWrapper::LoadKeys(String^ galKeys, String^ galKeysSingleStep, String^ relinKeys)
        {
            // Convert to std::string
            marshal_context marshalContext;
            string galString = marshalContext.marshal_as<std::string>(galKeys);
            string galSingleStepString = marshalContext.marshal_as<std::string>(galKeysSingleStep);
            string relinString = marshalContext.marshal_as<std::string>(relinKeys);

            // Free the memory if already set
            if (m_galKeys) { free(m_galKeys); }
            if (m_galKeysSingleStep) { free(m_galKeysSingleStep); }
            if (m_relinKeys) { free(m_relinKeys); }

            // Make new instances of the keys
            m_galKeys = new GaloisKeys();
            m_galKeysSingleStep = new GaloisKeys();
            m_relinKeys = new RelinKeys();

            // Load the keys
            ToSealObject<GaloisKeys>(galString, *m_galKeys, true);
            ToSealObject<GaloisKeys>(galSingleStepString, *m_galKeysSingleStep, true);
            ToSealObject<RelinKeys>(relinString, *m_relinKeys, true);
        }

        String^ SEALWrapper::Encrypt(List<double>^ values)
        {
            // Convert the List to a vector
            vector<double> input;
            for (int i = 0; i < values->Count; i++)
            {
                input.push_back(values[i]);
            }

            // Encrypt
            Plaintext plain;
            m_encoder->encode(input, m_scale, plain);
            Ciphertext encrypted;
            m_encryptor->encrypt(plain, encrypted);

            // Convert to base64 and return
            string cipherBase64 = FromSealObject<Ciphertext>(encrypted, true);
            return gcnew String(cipherBase64.c_str());
        }

        List<double>^ SEALWrapper::Decrypt(String ^str)
        {
            // Convert to std::string
            marshal_context marshalContext;
            string cipherBase64 = marshalContext.marshal_as<std::string>(str);

            // Load the cipher
            Ciphertext ciphertext;
            ToSealObject<Ciphertext>(cipherBase64, ciphertext, true);

            // Decrypt
            Plaintext plaintext;
            m_decryptor->decrypt(ciphertext, plaintext);
            vector<double> result;
            m_encoder->decode(plaintext, result);

            // Covert to List and return
            List<double>^ resultList = gcnew List<double>((int)result.size());
            for (int i = 0; i < (int)result.size(); i++)
            {
                resultList->Add(result[i]);
            }
            return resultList;
        }
        
        bool SEALWrapper::AddCiphers(String^ CipherStr1, String^ CipherStr2)
        {
            // Convert to std::string
            msclr::interop::marshal_context marshalContext;
            std::string cipherStr1 = marshalContext.marshal_as<std::string>(CipherStr1);
            std::string cipherStr2 = marshalContext.marshal_as<std::string>(CipherStr2);

            // Convert the base64 strings to Ciphers
            Ciphertext Cipher1, Cipher2;
            ToSealObject<Ciphertext>(cipherStr1, Cipher1, true);
            ToSealObject<Ciphertext>(cipherStr2, Cipher2, true);

            // Make sure the input is the right format
            if (((int)Cipher1.poly_modulus_degree() != m_slotCount * 2) ||
                ((int)Cipher1.poly_modulus_degree() != (int)Cipher2.poly_modulus_degree()))
            {
                return false;
            }

            // Add the ciphertexts
            Ciphertext resultCipher;
            m_evaluator->add(Cipher1, Cipher2, resultCipher);

            // Set the base64 result and return true
            m_addedCipherBase64 = new string(FromSealObject<Ciphertext>(resultCipher, true));
            return true;
        }

        bool SEALWrapper::ComputeStatsCiphers(String^ CipherStr1, String^ CipherStr2, String^ SummaryMaskCipherStr, String^ CipherGyroStr)
        {
            // Convert to std::string
            msclr::interop::marshal_context marshalContext;
            std::string cipherStr1 = marshalContext.marshal_as<std::string>(CipherStr1);
            std::string cipherStr2 = marshalContext.marshal_as<std::string>(CipherStr2);
            std::string summaryMaskCipherStr = marshalContext.marshal_as<std::string>(SummaryMaskCipherStr);
            std::string cipherGyroStr = marshalContext.marshal_as<std::string>(CipherGyroStr);

            // Convert the base64 strings to Ciphers
            Ciphertext Cipher1, Cipher2, SummaryMaskCipher, CipherGyro;
            ToSealObject<Ciphertext>(cipherStr1, Cipher1, true);
            ToSealObject<Ciphertext>(cipherStr2, Cipher2, true);
            ToSealObject<Ciphertext>(summaryMaskCipherStr, SummaryMaskCipher, true);
            ToSealObject<Ciphertext>(cipherGyroStr, CipherGyro, true);

            // Make sure the input is the right format
            if (((int)Cipher1.poly_modulus_degree() != m_slotCount * 2) ||
                ((int)Cipher1.poly_modulus_degree() != (int)Cipher2.poly_modulus_degree()) ||
                ((int)Cipher1.poly_modulus_degree() != (int)SummaryMaskCipher.poly_modulus_degree()) ||
                ((int)Cipher1.poly_modulus_degree() != (int)CipherGyro.poly_modulus_degree()))
            {
                return false;
            }

            // Calculate the stats cipher
            Ciphertext statsCipher, summaryCipher;
            int slotCount = (int)Cipher1.poly_modulus_degree() / 2;
            
            // Machine learning section that multiplies gyroscope and accelerometer data
            const double tensorB[] = {
                0.41964226961135864,
                6.6290693283081055,
                -2.404352903366089,
                -0.024301817640662193,
                -0.17596858739852905,
                0.14117737114429474
            };
            vector<double> tensorBVector(tensorB, tensorB + 6);
            tensorBVector.resize(slotCount, 0);
            Plaintext tensorBPlain;
            Ciphertext tensorOutput = CipherGyro;
            m_encoder->encode(tensorBVector, m_scale, tensorBPlain);
            m_evaluator->multiply_plain_inplace(tensorOutput, tensorBPlain);
            Ciphertext tensorOutputCopy = tensorOutput;
            for(int i = 1; i < 6; ++i) {
                Ciphertext copy = tensorOutputCopy;
                m_evaluator->rotate_vector_inplace(copy, i, *m_galKeys );
                m_evaluator->add_inplace(tensorOutput, copy);
            }
            vector<double> tensorMaskVector(slotCount);
            tensorMaskVector[0] = 1;
            Plaintext tensorMaskPlain;
            m_encoder->encode(tensorMaskVector, m_scale, tensorMaskPlain);
            m_evaluator->multiply_plain_inplace(tensorOutput, tensorMaskPlain);

            // Calculate the distance differences
            Ciphertext firstCipherShifted, secondCipherShifted;

            // Rotate with single step key (smaller dbc)
            m_evaluator->rotate_vector(Cipher1, 1, *m_galKeysSingleStep, firstCipherShifted);
            m_evaluator->rotate_vector(Cipher2, 1, *m_galKeysSingleStep, secondCipherShifted);

            m_evaluator->sub_inplace(firstCipherShifted, Cipher1);
            m_evaluator->sub_inplace(secondCipherShifted, Cipher2);

            // At this point the firstCipherShifted, secondCipherShifted contain
            // the differences in the location ticks at the most of (slotCount / 2) - 1 ticks.
            // The rest is garbage, so remove the garbage before squaring:
            Plaintext maskPlain1FirstHalf, maskPlain1SecondHalf,
                maskPlain2, maskPlain3, maskPlain4, maskPlain5;

            vector<double> mask1FirstHalf((slotCount / 2) - 1, 1);
            mask1FirstHalf.resize(slotCount, 0);

            vector<double> mask1SecondHalf(slotCount / 2, 0);
            mask1SecondHalf.resize(slotCount - 1, 1);
            mask1SecondHalf.push_back(0);

            // Encode mask1First with scale m_scale_small
            m_encoder->encode(mask1FirstHalf, m_scale_small, maskPlain1FirstHalf);

            // Encode mask1Second with scale m_scale_small
            m_encoder->encode(mask1SecondHalf, m_scale_small, maskPlain1SecondHalf);

            // firstCipherShifted is now at m_scale * m_scale_small
            Ciphertext firstCipherShiftedSecondHalf;
            m_evaluator->multiply_plain(firstCipherShifted, maskPlain1SecondHalf,
                firstCipherShiftedSecondHalf);
            m_evaluator->multiply_plain_inplace(firstCipherShifted, maskPlain1FirstHalf);

            // secondCipherShifted is now at m_scale * m_scale_small
            m_evaluator->multiply_plain_inplace(secondCipherShifted, maskPlain1FirstHalf);

            // Scale in both now m_scale^2 * m_scale_small^2
            m_evaluator->square_inplace(firstCipherShifted);
            m_evaluator->square_inplace(firstCipherShiftedSecondHalf);
            m_evaluator->square_inplace(secondCipherShifted);

            // Now data is safe from relin and rotate
            m_evaluator->relinearize_inplace(firstCipherShifted, *m_relinKeys);
            m_evaluator->relinearize_inplace(firstCipherShiftedSecondHalf, *m_relinKeys);
            m_evaluator->relinearize_inplace(secondCipherShifted, *m_relinKeys);

            // Rotate with bigger dbc Galois keys
            m_evaluator->rotate_vector(firstCipherShiftedSecondHalf,
                slotCount / 2, *m_galKeys, statsCipher);

            // Scale at m_scale^2 * m_scale_small^2
            m_evaluator->add_inplace(statsCipher, firstCipherShifted);
            m_evaluator->add_inplace(statsCipher, secondCipherShifted);

            // Calculate the Total Distance Traveled
            // cipherBuffer, summedCipher now m_scale^2 * m_scale_small^2; at full primes
            Ciphertext cipherBuffer = statsCipher;
            Ciphertext summedCipher = statsCipher;

            for (int i = 1; i < slotCount; i <<= 1)
            {
                m_evaluator->rotate_vector(summedCipher, i, *m_galKeys, cipherBuffer);
                m_evaluator->add_inplace(summedCipher, cipherBuffer);
            }

            // Rescale to m_scale * m_scale_small^2; down one prime
            m_evaluator->rescale_to_next_inplace(statsCipher);
            m_evaluator->rescale_to_next_inplace(summedCipher);

            vector<double> mask2(slotCount / 2, 0);
            mask2.resize(slotCount, 1);

            // Encode mask with scale m_scale_small
            m_encoder->encode(mask2, summedCipher.parms_id(), m_scale_small, maskPlain2);

            // Lift scale of statsCipher up to m_scale * m_scale_small^3
            Plaintext scaleLifter;
            m_encoder->encode(1.0, statsCipher.parms_id(), m_scale_small, scaleLifter);
            m_evaluator->multiply_plain_inplace(statsCipher, scaleLifter);

            // summedCipher scale is still m_scale * m_scale_small^3; down one prime
            m_evaluator->multiply_plain_inplace(summedCipher, maskPlain2);

            // Now can add; both at scale m_scale * m_scale_small^3 and down one prime
            m_evaluator->add_inplace(statsCipher, summedCipher);

            // summaryCipher at scale m_scale * m_scale_small^3; down one prime
            summaryCipher = summedCipher;

            // At this point the values after slotCount / 2 are all the total distance and 
            // 0 to (slotCount / 2) - 1 are the distance deltas.
            // The only available slot not used is (slotCount / 2) - 1, populate with total time:

            // Cipher2 is at original scale m_scale and full parameters
            // Lift scale up to m_scale * m_scale_small^2
            m_encoder->encode(1.0, Cipher2.parms_id(), m_scale_small * m_scale_small, scaleLifter);
            m_evaluator->multiply_plain_inplace(Cipher2, scaleLifter);

            m_evaluator->rotate_vector(Cipher2, slotCount / 2, *m_galKeys, cipherBuffer);

            // cipherBuffer is at original scale m_scale and full parameters
            m_evaluator->rotate_vector(cipherBuffer, slotCount / 2 + 1, *m_galKeys, Cipher2);

            // No problem with subtraction
            m_evaluator->sub_inplace(cipherBuffer, Cipher2);

            vector<double> mask3(slotCount, 0);
            mask3[(slotCount / 2) - 1] = 1;

            // Encode mask with scale m_scale * m_scale_small
            m_encoder->encode(mask3, cipherBuffer.parms_id(), m_scale * m_scale_small, maskPlain3);

            // Scale in cipherBuffer now m_scale * m_scale_small^3
            m_evaluator->multiply_plain_inplace(cipherBuffer, maskPlain3);

            // Rescale cipherBuffer to scale m_scale * m_scale_small^3; down one prime
            m_evaluator->rescale_to_next_inplace(cipherBuffer);

            // Now addition works; both at scale m_scale * m_scale_small^3 and down one prime
            m_evaluator->add_inplace(statsCipher, cipherBuffer);

            // Mask the total distance part of the stats with the meta data mask
            // disSummaryMask at full parameters and scale m_scale
            Ciphertext disSummaryMask = SummaryMaskCipher;
            vector<double> mask4(slotCount / 2, 0);
            mask4.resize(slotCount, 1);

            // Encode mask with scale m_scale_small
            m_encoder->encode(mask4, disSummaryMask.parms_id(), m_scale_small, maskPlain4);

            // Scale m_scale * m_scale_small
            m_evaluator->multiply_plain_inplace(disSummaryMask, maskPlain4);

            vector<double> mask5(slotCount / 2, 1);
            mask4.resize(slotCount, 0);

            // Encode with same parms and scale as disSummaryMask (m_scale * m_scale_small)
            m_encoder->encode(mask5, disSummaryMask.parms_id(), disSummaryMask.scale(), maskPlain5);

            m_evaluator->add_plain_inplace(disSummaryMask, maskPlain5);

            // Rescale to scale m_scale_small; down one prime
            m_evaluator->rescale_to_next_inplace(disSummaryMask);

            // Scale up to m_scale * m_scale_small^4; down one prime
            m_evaluator->multiply_inplace(statsCipher, disSummaryMask);

            // No problem relinearizing
            m_evaluator->relinearize_inplace(statsCipher, *m_relinKeys);

            // Rescale down
            m_evaluator->rescale_to_next_inplace(statsCipher);

            // Create the summary cipher
            // Scale m_scale * m_scale_small^3 and down one prime
            Ciphertext shiftedCipher = cipherBuffer;
            for (int i = 1; i < (slotCount / 2); i *= 2)
            {
                m_evaluator->rotate_vector(cipherBuffer, i, *m_galKeys, shiftedCipher);
                m_evaluator->add_inplace(cipherBuffer, shiftedCipher);
            }

            // Scale at m_scale * m_scale_small^3; down one prime
            m_evaluator->add_inplace(summaryCipher, cipherBuffer);

            // SummaryMaskCipher at full parameters and scale m_scale;
            // need to switch parameters down (no rescaling)
            m_evaluator->mod_switch_to_next_inplace(SummaryMaskCipher);

            // Scale m_scale^2 * m_scale_small^3; one prime down
            m_evaluator->multiply_inplace(summaryCipher, SummaryMaskCipher);

            // No problem with relinearize; no rescaling before this
            m_evaluator->relinearize_inplace(summaryCipher, *m_relinKeys);

            // Rescale down
            m_evaluator->rescale_to_next_inplace(summaryCipher);

            // Set the base64 results and return true
            m_statsCipherBase64 = new string(FromSealObject<Ciphertext>(statsCipher, true));
            m_summaryCipherBase64 = new string(FromSealObject<Ciphertext>(summaryCipher, true));
            m_mlCipherBase64 = new string(FromSealObject<Ciphertext>(tensorOutput, true));
            return true;
        }

        template<class T>
        void SEALWrapper::ToSealObject(string input, T &output, bool base64)
        {
            if (base64)
            {
                input = base64_decode(input);
            }
            stringstream stream(input);
            output.unsafe_load(stream);
        }

        template<class T>
        string SEALWrapper::FromSealObject(T input, bool base64)
        {
            stringstream stream;
            input.save(stream);
            string encryptedString = stream.str();
            if (base64)
            {
                encryptedString = base64_encode(reinterpret_cast<const uint8_t*>(encryptedString.c_str()), encryptedString.size());
            }
            return encryptedString;
        }
    }
}
