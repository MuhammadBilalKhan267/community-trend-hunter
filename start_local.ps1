param(
    [int]$Port = 11435,
    [string]$RuntimePath,
    [string]$ModelsPath
)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskCache = Join-Path $taskRoot '.cache'
New-Item -ItemType Directory -Path $taskCache -Force | Out-Null
$taskEndpoint = "http://127.0.0.1:$Port"
try {
    $taskVersion = Invoke-RestMethod -Uri "$taskEndpoint/api/version" -TimeoutSec 2
    Write-Output "Local server already responds at $taskEndpoint (version $($taskVersion.version))."
    Write-Output 'Existing server settings have not been inspected or changed.'
    exit 0
} catch {
    # No responding endpoint: start a separate task-local helper.
}
$taskOllama = if ($RuntimePath) { (Resolve-Path -LiteralPath $RuntimePath).Path } else { (Get-Command ollama -ErrorAction Stop).Source }
$taskOldHost = $env:OLLAMA_HOST
$taskOldAttention = $env:OLLAMA_FLASH_ATTENTION
$taskOldCloud = $env:OLLAMA_NO_CLOUD
$taskOldModels = $env:OLLAMA_MODELS
$taskOldLoaded = $env:OLLAMA_MAX_LOADED_MODELS
$taskOldParallel = $env:OLLAMA_NUM_PARALLEL
try {
    $env:OLLAMA_HOST = "127.0.0.1:$Port"
    $env:OLLAMA_FLASH_ATTENTION = 'false'
    $env:OLLAMA_NO_CLOUD = '1'
    $env:OLLAMA_MAX_LOADED_MODELS = '1'
    $env:OLLAMA_NUM_PARALLEL = '1'
    if ($ModelsPath) {
        New-Item -ItemType Directory -Path $ModelsPath -Force | Out-Null
        $env:OLLAMA_MODELS = (Resolve-Path -LiteralPath $ModelsPath).Path
    }
    $taskServer = Start-Process -FilePath $taskOllama -ArgumentList 'serve' -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskCache "ollama-$Port.out.log") -RedirectStandardError (Join-Path $taskCache "ollama-$Port.err.log") -PassThru
    $taskServer.Id | Set-Content -LiteralPath (Join-Path $taskCache "ollama-$Port.pid")
    Write-Output "Started task-local Ollama PID $($taskServer.Id) at $taskEndpoint. Logs are in $taskCache."
    Write-Output "Run the detector with --ollama-url $taskEndpoint ."
} finally {
    $env:OLLAMA_HOST = $taskOldHost
    $env:OLLAMA_FLASH_ATTENTION = $taskOldAttention
    $env:OLLAMA_NO_CLOUD = $taskOldCloud
    $env:OLLAMA_MODELS = $taskOldModels
    $env:OLLAMA_MAX_LOADED_MODELS = $taskOldLoaded
    $env:OLLAMA_NUM_PARALLEL = $taskOldParallel
}
