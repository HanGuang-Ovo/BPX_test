[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$OutputDirectory = "wheelhouse",

    [switch]$DisableProxy
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Test-Path Variable:\PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

if ([System.Environment]::OSVersion.Platform -ne [System.PlatformID]::Win32NT) {
    [Console]::Error.WriteLine("Error: this script only supports Windows.")
    exit 1
}

$RootDirectory = Split-Path -Parent $PSScriptRoot
$VenvDirectory = Join-Path $RootDirectory ".venv-cibw-win"
$VenvPython = Join-Path $VenvDirectory "Scripts\python.exe"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    [Console]::Error.WriteLine(@"
Error: uv is required but was not found in PATH.
Install uv first: https://docs.astral.sh/uv/getting-started/installation/
"@)
    exit 127
}

$ProxyVariableNames = @(
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "PIP_PROXY",
    "NO_PROXY"
)
$SavedProxyValues = @{}

if ($DisableProxy) {
    foreach ($Name in $ProxyVariableNames) {
        $SavedProxyValues[$Name] = [System.Environment]::GetEnvironmentVariable(
            $Name,
            [System.EnvironmentVariableTarget]::Process
        )
        [System.Environment]::SetEnvironmentVariable(
            $Name,
            $null,
            [System.EnvironmentVariableTarget]::Process
        )
    }

    # Python falls back to the Windows Internet Options proxy when the proxy
    # variables are absent. NO_PROXY=* forces pip and requests to connect
    # directly even when a system proxy is enabled.
    [System.Environment]::SetEnvironmentVariable(
        "NO_PROXY",
        "*",
        [System.EnvironmentVariableTarget]::Process
    )
    Write-Host "Proxy use is disabled for this build."
}

$BuildExitCode = 1
try {
    $CreateVenv = -not (Test-Path -LiteralPath $VenvPython -PathType Leaf)
    if (-not $CreateVenv) {
        & $VenvPython -c "import sys; raise SystemExit(sys.version_info[:2] != (3, 11))"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Recreating .venv-cibw-win because it does not use Python 3.11."
            $CreateVenv = $true
        }
        else {
            & $VenvPython -m pip --version *> $null
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Recreating .venv-cibw-win because pip is not installed."
                $CreateVenv = $true
            }
        }
    }

    if ($CreateVenv) {
        & uv venv --clear --python 3.11 --seed $VenvDirectory
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    else {
        Write-Host "Reusing Python 3.11 environment at $VenvDirectory"
    }

    Push-Location $RootDirectory
    try {
        & $VenvPython scripts/build_wheels.py --cibuildwheel --out-dir $OutputDirectory
        $BuildExitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($DisableProxy) {
        foreach ($Name in $ProxyVariableNames) {
            [System.Environment]::SetEnvironmentVariable(
                $Name,
                $SavedProxyValues[$Name],
                [System.EnvironmentVariableTarget]::Process
            )
        }
    }
}

exit $BuildExitCode
