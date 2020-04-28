// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>
#include <seal/seal.h>

/**
 * Handles all the complex SEAL operations for encryption and decryption and exposes a simple to use
 * interface.
 */
class CryptoContext
{
public:
    /**
     * The constructor which allocates all the resources for SEAL
     *
     * @param parms EncryptionParameters to use for SEAL. createCryptoContext sets these already,
     * but if the client wishes to use their own then they can call the constructor directly and
     * provide their own parameters.
     *
     * @param parms The parameters for SEAL to build off with
     * @param fileStorageDirectory The directory for keys to be saved out to
     */
    CryptoContext(const seal::EncryptionParameters &parms);
    ~CryptoContext();

    /**
     * Loads the public and secret keys which get saved from a call to generateKeys. These paths
     * must match the ones passed into generateKeys, and must be called before any encryption or
     * decryption take place. This is meant for subsequent runs of the app to speed up loading of
     * the public and private keys as key generation is very slow and they need to persist for
     * future runs. If this call fails that means either the keys don't exist, or they failed to
     * load into SEAL and must be (re)generated.
     *
     * @param publicKeyOutputPath The path for the public key to load from.
     * @param secretKeyOutputPath The path for the secret key to load from.
     *
     * @return true if both keys load successfully, false otherwise
     */
    bool loadLocalKeys(const std::string &publicKeyPath, const std::string &secretKeyPath);

    /**
     * Generate a new set of keys and saves them to files.
     *
     * @param publicKeyOutputPath The path for the public key to save to.
     * @param secretKeyOutputPath The path for the secret key to save to.
     * @param galoisKeyOutputPath The path for the Galois key to save to.
     * @param galoisSingleStepKeyOutputPath The path for the Galois single step key to save to.
     * @param relinearizeKeyOutputPath The path for the relinearize key to save to.
     */
    void generateKeys(
            const std::string &publicKeyPath,
            const std::string &secretKeyPath,
            const std::string &galoisKeyPath,
            const std::string &galoisSingleStepKeyPath,
            const std::string &relinearizeKeyPath);

    /**
     * Encrypts a vector of doubles and outputs a Base64 string
     *
     * @param input A vector of doubles to encrypt
     */
    std::string encrypt(const std::vector<double> &input);

    /**
     * Decrypts a Base64 encoded encrypted string and outputs a vector of doubles.
     *
     * @param input The Base64 encoded encrypted string
     */
    std::vector<double> decrypt(const std::string &input);

private:
    seal::EncryptionParameters m_parms;
    uint64_t m_scale;
    std::shared_ptr<seal::SEALContext> m_context;
    seal::PublicKey m_public_key;
    seal::SecretKey m_secret_key;
    seal::CKKSEncoder m_encoder;
};

/*
 * Creates and returns a CryptoContext. It handles all the encryption parameter setting without the
 * client needing to know how it's all done under the hood, not to mention AsureRun's use case
 * warranted a custom way of key generation.
 *
 * @param fileStorageDirectory The directory for keys to be saved out to
 * @param polyModulus The poly modulus for creating the SEAL context
 */
CryptoContext *createCryptoContext(const std::string &fileStorageDirectory, int polyModulus);

/*
 * Releases the CryptoContext's resources. C++ delete could also be called on it, but this is just
 * here as a counterpart to createCryptoContext.
 *
 * @param context The CryptoContext to release. It's also possible to just call delete on it, as
 * that's all that happens in this function anyway.
 */
void releaseCryptoContext(CryptoContext *context);
