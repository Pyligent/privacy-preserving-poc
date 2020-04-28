// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "CryptoContext.h"
#include "base64.h"
#include <sstream>
#include <fstream>

using namespace seal;
using namespace std;

/**
 * Takes a SEAL object and encodes it to a Base64 string
 *
 * @param object The SEAL object to encode
 */
template <typename T>
static string encodeSealToBase64(const T &object)
{
    ostringstream ss;
    object.save(ss);
    return base64_encode(ss.str());
}

/**
 * Takes a SEAL key and saves it out to the device's file system
 *
 * @param filePath Path to save the file
 * @param key The key to save out
 */
template <typename T>
static void saveToFile(const string &filePath, T &key)
{
    string keyString = encodeSealToBase64(key);
    ofstream saveFile(filePath, ios_base::binary);
    saveFile.write(keyString.c_str(), keyString.size());
}

/**
 * Loads a SEAL key from the device's file system
 *
 * @param filePath Path to load the file from
 * @param key The key to load out into
 */
template <typename T>
static bool loadFromFile(const string &filePath, T &key)
{
    ifstream file(filePath, ios_base::binary);
    if(file.is_open())
    {
        stringstream ss;
        ss << file.rdbuf();
        string keyString = base64_decode(ss.str());
        ss.str(keyString);
        key.unsafe_load(ss);
        return true;
    }
    return false;
}

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

CryptoContext::CryptoContext(const EncryptionParameters &parms)
        : m_parms(parms),
          m_scale(m_parms.coeff_modulus().back().value()),
          m_context(SEALContext::Create(m_parms)),
          m_encoder(m_context)
{
}

CryptoContext::~CryptoContext() {}

bool CryptoContext::loadLocalKeys(const std::string &publicKeyPath, const std::string &secretKeyPath)
{
    //if either public or secret key cannot be loaded all keys must be recreated
    return loadFromFile(publicKeyPath, m_public_key) && loadFromFile(secretKeyPath, m_secret_key);
}

void CryptoContext::generateKeys(
        const std::string &publicKeyOutputPath,
        const std::string &secretKeyOutputPath,
        const std::string &galoisKeyOutputPath,
        const std::string &galoisSingleStepKeyOutputPath,
        const std::string &relinearizeKeyOutputPath)
{
    KeyGenerator keygen(m_context);

    m_public_key = keygen.public_key();
    saveToFile(publicKeyOutputPath, m_public_key);

    m_secret_key = keygen.secret_key();
    saveToFile(secretKeyOutputPath, m_secret_key);

    GaloisKeys gal_keys = keygen.galois_keys(dbc_max(), half_size_gal_keys(m_context));
    saveToFile(galoisKeyOutputPath, gal_keys);

    vector<uint64_t> galoisElts = { uint64_t(3) };
    GaloisKeys gal_single_step_keys = keygen.galois_keys(15, galoisElts);
    saveToFile(galoisSingleStepKeyOutputPath, gal_single_step_keys);

    RelinKeys ev_keys = keygen.relin_keys(30);
    saveToFile(relinearizeKeyOutputPath, ev_keys);
}

string CryptoContext::encrypt(const vector<double> &input)
{
    Plaintext plain;
    m_encoder.encode(input, m_scale, plain);
    Ciphertext encrypted;
    Encryptor encryptor(m_context, m_public_key);
    encryptor.encrypt(plain, encrypted);
    return encodeSealToBase64(encrypted);
}

vector<double> CryptoContext::decrypt(const string &input)
{
    size_t slots = input.size() / sizeof(double);
    string decoded = base64_decode(input);
    stringstream stream;
    stream.write(decoded.data(), decoded.size());
    Decryptor decryptor(m_context, m_secret_key);
    Plaintext plainOutput;
    Ciphertext cipher;
    cipher.unsafe_load(stream);
    decryptor.decrypt(cipher, plainOutput);
    vector<double> realOutput;
    m_encoder.decode(plainOutput, realOutput);
    return realOutput;
}

CryptoContext *createCryptoContext(const string &fileStorageDirectory, int polyModulus)
{
    // The coeff_modulus parameter below is secure up to polyModulus == 8192.
    if (polyModulus > 8192)
    {
        throw invalid_argument("insecure parameters");
    }

    EncryptionParameters parms(scheme_type::CKKS);
    parms.set_poly_modulus_degree(polyModulus);
    parms.set_coeff_modulus({
        0xffffffffffc0001, // 60 bits
        0x7fffffffffcc001, // 59 bits
        0x7fffffffffa4001, // 59 bits
        0xffffe80001       // 40 bits
    });
    CryptoContext *context = new CryptoContext(parms);
    return context;
}

void releaseCryptoContext(CryptoContext *context)
{
    delete context;
}
