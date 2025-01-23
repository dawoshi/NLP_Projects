/* Copyright 2023, The GenC Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License
==============================================================================*/
#include "third_party/llama.cpp/interop/backends/java/open_ai_client_jni.h"

#include <jni.h>

#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "third_party/llama.cpp/java/src/jni/jni_utils.h"

namespace genc {

std::string CallOpenAiClient(JavaVM* jvm, jobject open_ai_client,
                             std::string model_config, std::string request) {
  CHECK(jvm != nullptr) << "JVM is null";
  JNIEnv* env = GetJniEnv(jvm);
  if (env == nullptr) {
    LOG(ERROR) << "Couldn't get JNI env";
    return "";
  }

  jclass openAiClientClass = env->GetObjectClass(open_ai_client);
  if (openAiClientClass == nullptr) {
    ThrowRuntimeException("Couldn't retrieve Open AI client class from JNI",
                          env);
  }
  jmethodID callMethodId =
      env->GetMethodID(openAiClientClass, "call", "([B[B)Ljava/lang/String;");

  if (callMethodId == 0) {
    ThrowRuntimeException("Couldn't retrieve method id from JNI", env);
  }

  jbyteArray jmodel_config = GetJbyteArrayFromString(env, model_config);
  jbyteArray jrequest = GetJbyteArrayFromString(env, request);
  jobject response_string = env->CallObjectMethod(open_ai_client, callMethodId,
                                                  jmodel_config, jrequest);
  std::string response = GetString(env, (jstring)response_string);
  env->DeleteLocalRef(response_string);
  env->DeleteLocalRef(jrequest);
  env->DeleteLocalRef(jmodel_config);
  jvm->DetachCurrentThread();
  return response;
}

}  // namespace genc
