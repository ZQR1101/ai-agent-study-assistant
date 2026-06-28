[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [switch]$SkipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$backendUrl = "http://127.0.0.1:8000"
$frontendUrl = "http://127.0.0.1:5500"

function Write-Step([string]$Message) {
    Write-Host "[AI Study Assistant] $Message" -ForegroundColor Cyan
}

function Resolve-CommandPath([string]$Name, [string[]]$Candidates) {
    $command = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $command) {
        return $command.Source
    }

    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    return $null
}

function Resolve-NodePath {
    $candidates = @(
        (Join-Path $env:ProgramFiles "nodejs\node.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\nodejs\node.exe"),
        (Join-Path $env:USERPROFILE "scoop\apps\nodejs\current\node.exe"),
        (Join-Path $env:APPDATA "nvm\current\node.exe")
    )

    $codexRuntimeRoot = Join-Path $env:USERPROFILE ".cache\codex-runtimes"
    if (Test-Path -LiteralPath $codexRuntimeRoot) {
        $runtimeNodes = Get-ChildItem -LiteralPath $codexRuntimeRoot -Directory -ErrorAction SilentlyContinue |
            ForEach-Object { Join-Path $_.FullName "dependencies\node\bin\node.exe" }
        $candidates += $runtimeNodes
    }

    return Resolve-CommandPath "node.exe" $candidates
}

function Test-Port([int]$Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $connection = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        return $connection.AsyncWaitHandle.WaitOne(300, $false) -and $client.Connected
    }
    catch {
        return $false
    }
    finally {
        $client.Close()
    }
}

function Test-Backend {
    try {
        $response = Invoke-RestMethod -Uri "$backendUrl/health" -TimeoutSec 3
        return $response.status -eq "ok"
    }
    catch {
        return $false
    }
}

function Test-Frontend {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $frontendUrl -TimeoutSec 3
        # Match an ASCII-only marker so Windows PowerShell 5.1 response decoding
        # cannot turn a healthy UTF-8 page into a false negative.
        return $response.StatusCode -eq 200 -and $response.Content -match '<div id="root"></div>'
    }
    catch {
        return $false
    }
}

function Wait-ForService([string]$Name, [scriptblock]$HealthCheck, [System.Diagnostics.Process]$Process, [int]$TimeoutSeconds = 90) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (& $HealthCheck) {
            Write-Host "  OK  $Name 已就绪" -ForegroundColor Green
            return
        }
        if ($null -ne $Process -and $Process.HasExited) {
            throw "$Name 启动进程已退出，请查看对应的命令行窗口。"
        }
        Start-Sleep -Milliseconds 800
    }
    throw "等待 $Name 启动超时，请查看对应的命令行窗口。"
}

try {
    Set-Location $projectRoot
    Write-Step "正在检查运行环境..."

    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $venvPython)) {
        if ($SkipInstall) {
            throw "未找到 .venv。请去掉 -SkipInstall 后重试，或先手动创建虚拟环境。"
        }

        $basePython = Resolve-CommandPath "python.exe" @()
        if (-not $basePython) {
            $basePython = Resolve-CommandPath "py.exe" @()
        }
        if (-not $basePython) {
            throw "未找到 Python。请安装 Python 3.10+，安装时勾选 Add Python to PATH。"
        }

        Write-Step "首次运行：正在创建 Python 虚拟环境..."
        & $basePython -m venv (Join-Path $projectRoot ".venv")
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $venvPython)) {
            throw "Python 虚拟环境创建失败。"
        }
    }
    $pythonPath = (Resolve-Path -LiteralPath $venvPython).Path

    & $pythonPath -c "import fastapi, uvicorn" 2>$null
    if ($LASTEXITCODE -ne 0) {
        if ($SkipInstall) {
            throw "Python 依赖不完整。请运行 pip install -r requirements.txt。"
        }
        Write-Step "首次运行：正在安装 Python 依赖，这可能需要几分钟..."
        & $pythonPath -m pip install -r (Join-Path $projectRoot "requirements.txt")
        if ($LASTEXITCODE -ne 0) {
            throw "Python 依赖安装失败。"
        }
    }

    $nodePath = Resolve-NodePath
    if (-not $nodePath) {
        throw "未找到 Node.js。请安装 Node.js 18+ 后重新双击 start.bat。"
    }

    $viteEntry = Join-Path $projectRoot "node_modules\vite\bin\vite.js"
    if (-not (Test-Path -LiteralPath $viteEntry)) {
        if ($SkipInstall) {
            throw "前端依赖不完整。请运行 npm ci。"
        }

        $npmCandidates = @(
            (Join-Path (Split-Path -Parent $nodePath) "npm.cmd"),
            (Join-Path $env:ProgramFiles "nodejs\npm.cmd")
        )
        $npmPath = Resolve-CommandPath "npm.cmd" $npmCandidates
        if (-not $npmPath) {
            throw "缺少前端依赖且未找到 npm。请重新安装完整版 Node.js 后运行 npm ci。"
        }

        Write-Step "首次运行：正在安装前端依赖..."
        & $npmPath ci
        if ($LASTEXITCODE -ne 0) {
            throw "前端依赖安装失败。"
        }
    }

    $envFile = Join-Path $projectRoot ".env"
    if (-not (Test-Path -LiteralPath $envFile)) {
        Copy-Item -LiteralPath (Join-Path $projectRoot ".env.example") -Destination $envFile
        Write-Host "  已根据 .env.example 创建 .env；使用在线模型前请填写 API Key。" -ForegroundColor Yellow
    }

    $backendRunning = Test-Backend
    if (-not $backendRunning -and (Test-Port 8000)) {
        throw "端口 8000 已被其他程序占用。请关闭占用程序后重试。"
    }

    $frontendRunning = Test-Frontend
    if (-not $frontendRunning -and (Test-Port 5500)) {
        throw "端口 5500 已被其他程序占用。请关闭占用程序后重试。"
    }

    $backendProcess = $null
    if ($backendRunning) {
        Write-Host "  OK  后端已在运行" -ForegroundColor Green
    }
    else {
        Write-Step "正在启动后端..."
        $env:AI_STUDY_PYTHON = $pythonPath
        $backendProcess = Start-Process -FilePath (Join-Path $PSScriptRoot "run-backend.cmd") -WorkingDirectory $projectRoot -PassThru
    }

    $frontendProcess = $null
    if ($frontendRunning) {
        Write-Host "  OK  前端已在运行" -ForegroundColor Green
    }
    else {
        Write-Step "正在启动前端..."
        $env:AI_STUDY_NODE = $nodePath
        $frontendProcess = Start-Process -FilePath (Join-Path $PSScriptRoot "run-frontend.cmd") -WorkingDirectory $projectRoot -PassThru
    }

    if (-not $backendRunning) {
        Wait-ForService "后端" ${function:Test-Backend} $backendProcess
    }
    if (-not $frontendRunning) {
        Wait-ForService "前端" ${function:Test-Frontend} $frontendProcess
    }

    Write-Host ""
    Write-Host "启动成功：$frontendUrl" -ForegroundColor Green
    Write-Host "API 文档：$backendUrl/docs"
    Write-Host "关闭两个服务窗口即可停止项目。"

    if (-not $NoBrowser) {
        Start-Process $frontendUrl
    }
}
catch {
    Write-Host ""
    Write-Host "启动失败：$($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
