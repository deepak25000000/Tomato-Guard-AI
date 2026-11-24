allprojects {
    repositories {
        google()
        mavenCentral()
    }
    configurations.all {
        resolutionStrategy {
            force("org.tensorflow:tensorflow-lite:2.16.1")
            force("org.tensorflow:tensorflow-lite-api:2.16.1")
            force("org.tensorflow:tensorflow-lite-gpu:2.16.1")
            force("org.tensorflow:tensorflow-lite-gpu-api:2.16.1")
            force("org.tensorflow:tensorflow-lite-support:0.4.4")
        }
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}
// Fix JVM target mismatch and compileSdk for plugin subprojects (e.g. tflite_flutter).
subprojects {
    tasks.withType<JavaCompile>().configureEach {
        sourceCompatibility = "17"
        targetCompatibility = "17"
    }
    tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        }
    }
    // compileSdk override for plugins handled by patching plugin if needed; avoid late configuration.
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
