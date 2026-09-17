param(
    [Parameter(Mandatory)][string]$Archive,
    [switch]$RequireGpu
)
$ErrorActionPreference = 'Stop'
$root = Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())
$savedPath = $env:PATH
New-Item -ItemType Directory $root | Out-Null
function Invoke-Worker([string]$Exe, [string]$Argument) {
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $Exe
    $info.Arguments = $Argument
    $info.WorkingDirectory = Split-Path $Exe
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $process = [Diagnostics.Process]::Start($info)
    $stdout = $process.StandardOutput.ReadToEndAsync()
    $stderr = $process.StandardError.ReadToEndAsync()
    try {
        if (!$process.WaitForExit(15000)) { $process.Kill(); throw 'Worker probe exceeded 15 seconds' }
        return @{ Code = $process.ExitCode; Out = $stdout.GetAwaiter().GetResult(); Err = $stderr.GetAwaiter().GetResult() }
    } finally { $process.Dispose() }
}
try {
    $package = Join-Path $root 'package'
    Expand-Archive $Archive $package
    $exe = Join-Path $package 'ck-synapse-worker-cuda.exe'
    $manifest = Get-Content (Join-Path $package 'manifest.json') -Raw | ConvertFrom-Json
    if ((Get-FileHash $exe -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.worker_sha256) { throw 'Worker hash mismatch' }
    foreach ($entry in $manifest.runtime_files) {
        if ((Get-FileHash (Join-Path $package $entry.file) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw "Hash mismatch: $($entry.file)" }
    }
    $empty = Join-Path $root 'no-sidecars'
    New-Item -ItemType Directory $empty | Out-Null
    Copy-Item $exe $empty
    $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
    $isolatedExe = Join-Path $empty 'ck-synapse-worker-cuda.exe'
    $version = Invoke-Worker $isolatedExe '--version'
    if ($version.Code -ne 0 -or $version.Out -notmatch '^ck-synapse-worker-cuda ') { throw "No-DLL version failed: $($version.Err)" }
    $missing = Invoke-Worker $isolatedExe '--probe-floor'
    if ($missing.Code -eq 0 -or $missing.Err -notmatch 'cannot load CUDA library') { throw "Missing-DLL refusal failed: $($missing.Code) $($missing.Err)" }
    $present = Invoke-Worker $exe '--probe-floor'
    if ($present.Code -eq 0) {
        $floor = $present.Out | ConvertFrom-Json
        if ($floor.driver_api -le 0 -or $floor.compute_capability.major -le 0) { throw 'Invalid floor JSON' }
        if ($RequireGpu -and ($floor.driver_api -lt 12040 -or $floor.compute_capability.major -lt 7 -or ($floor.compute_capability.major -eq 7 -and $floor.compute_capability.minor -lt 5))) {
            throw 'GPU below owned-CUDA floor: driver API >= 12040 and compute capability >= 7.5 required'
        }
        Write-Output "PASS packaged GPU probe: $($present.Out.Trim())"
    } elseif ($RequireGpu) {
        throw "Packaged GPU probe failed: $($present.Code) $($present.Err)"
    } elseif ($present.Err -match 'cannot load CUDA library (cublas|cudart)' -or $present.Err -notmatch '(nvcuda.dll|cuInit|cuDeviceGet)') {
        throw "Packaged runtime resolution failed: $($present.Code) $($present.Err)"
    } else {
        Write-Output "GPU execution not available on this runner: $($present.Err.Trim())"
    }
    Write-Output 'PASS archive hashes, no-DLL version, missing-DLL refusal, side-by-side runtime resolution'
} finally {
    $env:PATH = $savedPath
    Remove-Item $root -Recurse -Force
}
