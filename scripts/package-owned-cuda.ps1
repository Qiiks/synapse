param(
    [Parameter(Mandatory)][string]$Worker,
    [Parameter(Mandatory)][string[]]$RuntimeComponents,
    [Parameter(Mandatory)][string]$Output
)
$ErrorActionPreference = 'Stop'
$stage = Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())
New-Item -ItemType Directory $stage | Out-Null
try {
    Copy-Item $Worker (Join-Path $stage 'ck-synapse-worker-cuda.exe')
    $entries = @()
    foreach ($component in $RuntimeComponents) {
        $root = (Resolve-Path $component).Path
        $dlls = @(Get-ChildItem $root -Recurse -File -Filter '*.dll')
        if (!$dlls.Count) { throw "No runtime DLLs in $root" }
        foreach ($file in $dlls) {
            $destination = Join-Path $stage $file.Name
            if (Test-Path $destination) { throw "Duplicate runtime filename: $($file.Name)" }
            Copy-Item $file.FullName $destination
            $entries += [ordered]@{
                file = $file.Name
                sha256 = (Get-FileHash $destination -Algorithm SHA256).Hash.ToLowerInvariant()
                component = Split-Path $root -Leaf
                source = $file.FullName.Substring($root.Length + 1).Replace('\', '/')
            }
        }
        $licenseDir = Join-Path $stage ('licenses/' + (Split-Path $root -Leaf))
        New-Item -ItemType Directory -Force $licenseDir | Out-Null
        $licenses = @(Get-ChildItem $root -Recurse -File | Where-Object { $_.Name -match '^(LICENSE|EULA|COPYING)' })
        if (!$licenses.Count) { throw "Missing redistribution license in $root" }
        foreach ($license in $licenses) { Copy-Item $license.FullName $licenseDir }
    }
    [ordered]@{
        schema = 1
        worker_sha256 = (Get-FileHash (Join-Path $stage 'ck-synapse-worker-cuda.exe') -Algorithm SHA256).Hash.ToLowerInvariant()
        runtime_files = $entries
        driver = 'System-installed NVIDIA driver; not bundled'
    } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $stage 'manifest.json') -Encoding UTF8
    Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $Output -Force
    Get-FileHash $Output -Algorithm SHA256
} finally { Remove-Item $stage -Recurse -Force }
