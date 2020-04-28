// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <jni.h>
#include "CryptoContext.h"

using namespace std;

extern "C" JNIEXPORT jlong JNICALL Java_com_microsoft_asurerun_model_ApplicationState_nativeCreateCryptoContext(JNIEnv *env, jobject, jstring fileStorageDirectory, jint polyModulus, jint scale)
{
    const char *directory = env->GetStringUTFChars(fileStorageDirectory, nullptr);
    CryptoContext *context = createCryptoContext(directory, polyModulus);
    env->ReleaseStringUTFChars(fileStorageDirectory, directory);
    return reinterpret_cast<long>(context);
}

extern "C" JNIEXPORT void JNICALL Java_com_microsoft_asurerun_model_ApplicationState_nativeReleaseCryptoContext(JNIEnv *env, jobject, jlong contextHandle)
{
    releaseCryptoContext(reinterpret_cast<CryptoContext*>(contextHandle));
}

extern "C" JNIEXPORT jstring JNICALL Java_com_microsoft_asurerun_model_ApplicationState_nativeEncrypt(JNIEnv *env, jobject, jlong contextHandle, jdoubleArray input)
{
    jsize inputLength = env->GetArrayLength(input);
    jdouble *rawArray = env->GetDoubleArrayElements(input, nullptr);
    vector<double> inputVector(rawArray, rawArray + inputLength);
    env->ReleaseDoubleArrayElements(input, rawArray, JNI_ABORT);

    CryptoContext *context = reinterpret_cast<CryptoContext*>(contextHandle);
    return env->NewStringUTF(context->encrypt(inputVector).c_str());
}

extern "C" JNIEXPORT jdoubleArray JNICALL Java_com_microsoft_asurerun_model_ApplicationState_nativeDecrypt(JNIEnv *env, jobject, jlong contextHandle, jstring input)
{
    const char *rawInput = env->GetStringUTFChars(input, nullptr);
    CryptoContext *context = reinterpret_cast<CryptoContext*>(contextHandle);
    vector<double> output = context->decrypt(rawInput);
    env->ReleaseStringUTFChars(input, rawInput);
    jdoubleArray javaOutput = env->NewDoubleArray(output.size());
    env->SetDoubleArrayRegion(javaOutput, 0, output.size(), output.data());

    return javaOutput;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_microsoft_asurerun_model_ApplicationState_nativeLoadLocalKeys(
        JNIEnv *env,
        jobject,
        jlong contextHandle,
        jstring publicKeyPath,
        jstring secretKeyPath)
{
    const char *publicKey = env->GetStringUTFChars(publicKeyPath, nullptr);
    const char *secretKey = env->GetStringUTFChars(secretKeyPath, nullptr);
    CryptoContext *context = reinterpret_cast<CryptoContext*>(contextHandle);
    bool result = context->loadLocalKeys(publicKey, secretKey);
    env->ReleaseStringUTFChars(publicKeyPath, publicKey);
    env->ReleaseStringUTFChars(secretKeyPath, secretKey);
    return result == true;
}

extern "C" JNIEXPORT void JNICALL Java_com_microsoft_asurerun_model_ApplicationState_nativeGenerateKeys(
    JNIEnv *env,
    jobject,
    jlong contextHandle,
    jstring publicKeyPath,
    jstring secretKeyPath,
    jstring galoisKeyPath,
    jstring galoisSingleStepKeyPath,
    jstring relinearizeKeyPath)
{
    const char *publicKey = env->GetStringUTFChars(publicKeyPath, nullptr);
    const char *secretKey = env->GetStringUTFChars(secretKeyPath, nullptr);
    const char *galoisKey = env->GetStringUTFChars(galoisKeyPath, nullptr);
    const char *galoisSingleStepKey = env->GetStringUTFChars(galoisSingleStepKeyPath, nullptr);
    const char *relinearizeKey = env->GetStringUTFChars(relinearizeKeyPath, nullptr);
    CryptoContext *context = reinterpret_cast<CryptoContext*>(contextHandle);
    context->generateKeys(publicKey, secretKey, galoisKey, galoisSingleStepKey, relinearizeKey);
    env->ReleaseStringUTFChars(publicKeyPath, publicKey);
    env->ReleaseStringUTFChars(secretKeyPath, secretKey);
    env->ReleaseStringUTFChars(galoisKeyPath, galoisKey);
    env->ReleaseStringUTFChars(galoisSingleStepKeyPath, galoisSingleStepKey);
    env->ReleaseStringUTFChars(relinearizeKeyPath, relinearizeKey);
}