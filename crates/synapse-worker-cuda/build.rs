fn main() {
    if std::env::var_os("CARGO_FEATURE_CUDA").is_some()
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
    {
        // Link arguments on the engine rlib do not propagate to this executable.
        println!("cargo:rustc-link-lib=delayimp");
        // The CUDA 13 import libraries link cudart/cublas statically; cuBLASLt
        // is the remaining load-time DLL (verified with dumpbin /dependents).
        println!("cargo:rustc-link-arg=/DELAYLOAD:cublasLt64_13.dll");
    }
}
