plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("com.chaquo.python")
}

android {
    namespace = "com.arvoreneural.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.arvoreneural.app"
        minSdk = 26
        targetSdk = 34
        versionCode = 19
        versionName = "11.3"
        // Chaquopy exige abiFilters. Só arm64 (celulares atuais): deixa o APK bem menor.
        ndk { abiFilters += listOf("arm64-v8a") }
        // llama.cpp (C++) compilado junto: ver app/src/main/cpp/CMakeLists.txt
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DGGML_CPU_ARM_ARCH=armv8-a"
                )
            }
        }
    }
    // NDK e CMake instalados pelo workflow (passo "Instalar NDK e CMake").
    ndkVersion = "27.0.12077973"
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    // Chave fixa: o workflow grava arvore.jks na raiz a partir dos Secrets do GitHub.
    // Sem a chave, o build cai na assinatura de depuração (e o APK não atualiza por cima do antigo).
    val ksFile = rootProject.file("arvore.jks")
    signingConfigs {
        create("fixa") {
            if (ksFile.exists()) {
                storeFile = ksFile
                storePassword = System.getenv("KS_PASS")
                keyAlias = System.getenv("KS_ALIAS")
                keyPassword = System.getenv("KS_KEYPASS") ?: System.getenv("KS_PASS")
            }
        }
    }
    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = if (ksFile.exists()) signingConfigs.getByName("fixa") else signingConfigs.getByName("debug")
        }
        debug {
            if (ksFile.exists()) signingConfig = signingConfigs.getByName("fixa")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures { compose = true }
}

// Python dentro do app (Chaquopy) com NumPy; o interpretador do build precisa ser o python3 da máquina.
chaquopy {
    defaultConfig {
        version = "3.11"
        buildPython("python3")
        pip { install("numpy") }
    }
}

dependencies {
    implementation("com.google.ai.edge.litertlm:litertlm-android:latest.release")
    implementation(platform("androidx.compose:compose-bom:2024.06.00"))
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.0")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("org.mozilla:rhino:1.7.14")
}
