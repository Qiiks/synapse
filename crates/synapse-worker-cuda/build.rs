fn main() {
    if std::env::var_os("CARGO_FEATURE_CUDA").is_some()
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
    {
        let cuda_root = std::env::var_os("CUDA_HOME")
            .or_else(|| std::env::var_os("CUDA_PATH"))
            .map(std::path::PathBuf::from)
            .expect("Windows owned-CUDA packaging requires CUDA_HOME or CUDA_PATH");
        let header = cuda_root.join("include/cuda.h");
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rerun-if-changed={}", header.display());
        let contents =
            std::fs::read_to_string(&header).expect("cannot read CUDA toolkit include/cuda.h");
        let version = contents.lines().find_map(|line| {
            let mut fields = line.split_whitespace();
            (fields.next() == Some("#define") && fields.next() == Some("CUDA_VERSION"))
                .then(|| fields.next()?.parse::<u32>().ok())
                .flatten()
        });
        assert!(
            version.is_some_and(|version| version / 1000 == 13),
            "Windows owned-CUDA packaging requires CUDA 13; CUDA 12 DLL names are incompatible"
        );
        // Link arguments on the engine rlib do not propagate to this executable.
        println!("cargo:rustc-link-lib=delayimp");
        // The CUDA 13 import libraries link cudart/cublas statically; cuBLASLt
        // is the remaining load-time DLL (verified with dumpbin /dependents).
        println!("cargo:rustc-link-arg=/DELAYLOAD:cublasLt64_13.dll");
    }
}
