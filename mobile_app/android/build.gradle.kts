allprojects {
    repositories {
        google()
        mavenCentral()
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
// Some plugin modules (e.g. tflite_flutter) pin Java 11 in their own
// compileOptions while the Kotlin Gradle plugin defaults its jvmTarget to the
// host JDK (17 here), which Gradle now treats as a hard error
// ("Inconsistent JVM Target Compatibility Between Java and Kotlin Tasks").
// Force a single, consistent JVM target across every subproject so plugin
// modules build regardless of what they declare themselves. This must be
// registered before evaluationDependsOn(":app") below forces early
// evaluation of subprojects, or afterEvaluate throws on already-evaluated
// projects.
subprojects {
    afterEvaluate {
        extensions.findByType(com.android.build.gradle.BaseExtension::class.java)?.apply {
            compileOptions {
                sourceCompatibility = JavaVersion.VERSION_17
                targetCompatibility = JavaVersion.VERSION_17
            }
        }
        tasks.withType(org.jetbrains.kotlin.gradle.tasks.KotlinCompile::class.java).configureEach {
            compilerOptions {
                jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
            }
        }
    }
}

subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
