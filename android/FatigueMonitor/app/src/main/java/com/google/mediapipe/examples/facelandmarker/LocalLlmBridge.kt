package com.google.mediapipe.examples.facelandmarker

object LocalLlmBridge {
    private val loadError: Throwable? =
        runCatching {
            System.loadLibrary("fatigue_llm_jni")
        }.exceptionOrNull()

    private external fun nativeSmokeTest(): String
    private external fun nativeLlamaBackendSmokeTest(): String
    private external fun nativeModelLoadSmokeTest(modelPath: String): String
    private external fun nativeGenerateSmokeTest(modelPath: String, prompt: String): String
    private external fun nativeInitializeModel(modelPath: String): String
    private external fun nativeGenerate(prompt: String, maxNewTokens: Int): String
    private external fun nativeCancelGeneration()
    private external fun nativeReleaseModel(): String

    fun smokeTest(): String {
        loadError?.let { error ->
            return "H7.10.1 JNI_LOAD_ERROR ${error::class.java.simpleName}: ${error.message}"
        }

        return runCatching { nativeSmokeTest() }
            .getOrElse { error ->
                "H7.10.1 JNI_CALL_ERROR ${error::class.java.simpleName}: ${error.message}"
            }
    }

    fun llamaBackendSmokeTest(): String {
        loadError?.let { error ->
            return "H7.10.2 LLAMA_LOAD_ERROR ${error::class.java.simpleName}: ${error.message}"
        }

        return runCatching { nativeLlamaBackendSmokeTest() }
            .getOrElse { error ->
                "H7.10.2 LLAMA_CALL_ERROR ${error::class.java.simpleName}: ${error.message}"
            }
    }

    fun modelLoadSmokeTest(modelPath: String): String {
        loadError?.let { error ->
            return "H7.10.3 MODEL_LOAD_ERROR ${error::class.java.simpleName}: ${error.message}"
        }

        return runCatching { nativeModelLoadSmokeTest(modelPath) }
            .getOrElse { error ->
                "H7.10.3 MODEL_CALL_ERROR ${error::class.java.simpleName}: ${error.message}"
            }
    }

    fun generateSmokeTest(modelPath: String, prompt: String): String {
        loadError?.let { error ->
            return "H7.10.4 GENERATE_LOAD_ERROR ${error::class.java.simpleName}: ${error.message}"
        }

        return runCatching { nativeGenerateSmokeTest(modelPath, prompt) }
            .getOrElse { error ->
                "H7.10.4 GENERATE_CALL_ERROR ${error::class.java.simpleName}: ${error.message}"
            }
    }

    fun initializeModel(modelPath: String): String {
        loadError?.let { error ->
            return "H7.10.5 INIT_ERROR ${error::class.java.simpleName}: ${error.message}"
        }

        return runCatching { nativeInitializeModel(modelPath) }
            .getOrElse { error ->
                "H7.10.5 INIT_ERROR ${error::class.java.simpleName}: ${error.message}"
            }
    }

    fun generate(prompt: String, maxNewTokens: Int = 32): String {
        loadError?.let { error ->
            return "H7.10.5 GENERATE_ERROR ${error::class.java.simpleName}: ${error.message}"
        }

        return runCatching { nativeGenerate(prompt, maxNewTokens) }
            .getOrElse { error ->
                "H7.10.5 GENERATE_ERROR ${error::class.java.simpleName}: ${error.message}"
            }
    }

    fun cancelGeneration() {
        if (loadError == null) {
            runCatching { nativeCancelGeneration() }
        }
    }

    fun releaseModel(): String {
        loadError?.let { error ->
            return "H7.10.5 RELEASE_ERROR ${error::class.java.simpleName}: ${error.message}"
        }

        return runCatching { nativeReleaseModel() }
            .getOrElse { error ->
                "H7.10.5 RELEASE_ERROR ${error::class.java.simpleName}: ${error.message}"
            }
    }
}
