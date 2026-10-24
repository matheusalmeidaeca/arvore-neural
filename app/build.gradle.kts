plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.arvoreneural.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.arvoreneural.app"
        minSdk = 26
        targetSdk = 34
        versionCode = 4
        versionName = "7.2"
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
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }
    composeOptions { kotlinCompilerExtensionVersion = "1.5.14" }
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2024.06.00"))
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.0")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
}
