<#
.SYNOPSIS
  一次性跑完 8 个 condition 的实验矩阵，每个 condition 的结果落在各自的子文件夹里。

.DESCRIPTION
  矩阵里每个 condition 的 (structure, width, layers, personas) 是固定的，所以一个 condition
  = 一次 `python main.py` 调用（跑默认数据集里的全部 case）。各 condition 用独立的
  --out-dir，结果互不混淆，也方便后面按 condition 单独 evaluate。




  | id   | 命令参数                                                  | 干活 worker |
  |------|-----------------------------------------------------------|-------------|
  | N2   | decentralized --width 2 --personas generic                | 2           |
  | N4   | decentralized --width 4 --personas generic                | 4           |
  | N8   | decentralized --width 8 --personas generic                | 8           |
  | C1   | centralized   --layers 1 --width 2 --personas generic     | 2           |
  | C2   | centralized   --layers 2 --width 2 --personas generic     | 4           |
  | C3   | centralized   --layers 3 --width 2 --personas generic     | 8           |
  | N4-t | decentralized --width 4 --personas typed                  | 4           |
  | C2-t | centralized   --layers 2 --width 2 --personas typed       | 4           |

  总 run 数 = 8 condition × 数据集 task 数；当前默认 MAS 数据集是 600 task，即 4800 次 run。

.PARAMETER Workers
  每个 condition 内并发跑多少个 run（默认 8）。≈ key 数 × 每 key 能扛的并发。
.PARAMETER Root
  结果根目录（默认 "results\baseline"）。各 condition 写到 <Root>\<id>\。
.PARAMETER Visibility
  信息可见模式：scoped(默认) / full。
.PARAMETER Resume
  断点续跑：跳过各 condition 文件夹里已完成的 run（中途断了重跑这个脚本即可）。
.PARAMETER Only
  只跑指定的 condition（逗号分隔，如 -Only N4,C2）。不传则全跑。
.PARAMETER Defense
  Boundary-State Defense 开关：none(默认) / authorizer / chain / both，或逗号组合。
.PARAMETER LlmProvider
  raw worker 的 transport provider；OpenAI-compatible 中转统一传 openai。
.PARAMETER LlmModel
  raw worker 模型 ID。
.PARAMETER LlmBaseUrl
  raw worker 的 API root；可传中转 origin 或以 /v1 结尾的地址，不能传 /chat/completions。
.PARAMETER LlmApiKeysEnv
  保存一个或多个逗号分隔 key 的环境变量名；不会把 key 值写入命令或日志。

.EXAMPLE
  # 设好 key 后全量跑，16 并发，支持断点续跑
  $env:OPENAI_API_KEY = "sk-..."
  .\scripts\run_matrix.ps1 -Workers 16 -Resume

.EXAMPLE
  # 只跑去中心化的三档
  .\scripts\run_matrix.ps1 -Only N2,N4,N8
#>
param(
    [int]$Workers = 8,
    [string]$Root = "results\baseline",
    [ValidateSet("scoped", "full")][string]$Visibility = "scoped",
    [string]$Defense = "none",
    [string]$Python = "python",
    [string]$CaseId = "",
    [ValidateSet("openai", "deepseek", "google")][string]$LlmProvider = "",
    [string]$LlmModel = "",
    [string]$LlmBaseUrl = "",
    [string]$LlmApiKeysEnv = "",
    [switch]$Resume,
    [string[]]$Only
)

$ErrorActionPreference = "Stop"
# 中转/控制台编码兜底（避免 UnicodeEncodeError）
if (-not $env:PYTHONIOENCODING) { $env:PYTHONIOENCODING = "utf-8" }

# 每个 condition：id + 传给 main.py 的参数（--structure 之后的部分）
$conditions = @(
    @{ id = "N2";   args = @("--structure", "decentralized", "--width", "2", "--personas", "generic") }
    @{ id = "N4";   args = @("--structure", "decentralized", "--width", "4", "--personas", "generic") }
    @{ id = "N8";   args = @("--structure", "decentralized", "--width", "8", "--personas", "generic") }
    @{ id = "C1";   args = @("--structure", "centralized", "--layers", "1", "--width", "2", "--personas", "generic") }
    @{ id = "C2";   args = @("--structure", "centralized", "--layers", "2", "--width", "2", "--personas", "generic") }
    @{ id = "C3";   args = @("--structure", "centralized", "--layers", "3", "--width", "2", "--personas", "generic") }
    @{ id = "N4-t"; args = @("--structure", "decentralized", "--width", "4", "--personas", "typed") }
    @{ id = "C2-t"; args = @("--structure", "centralized", "--layers", "2", "--width", "2", "--personas", "typed") }
)

if ($Only) {
    # Force an array even when exactly one condition matches; otherwise a lone
    # hashtable reports Count=2 (its id/args keys) and progress says [1/2].
    $conditions = @($conditions | Where-Object { $Only -contains $_.id })
    if (-not $conditions) { throw "no matching condition in -Only $($Only -join ',')" }
}

$total = $conditions.Count
$i = 0
foreach ($c in $conditions) {
    $i++
    $outDir = Join-Path $Root $c.id
    $cmdArgs = @("main.py") + $c.args + @("--visibility", $Visibility, "--out-dir", $outDir, "--workers", $Workers)
    if ($Defense -and $Defense -ne "none") { $cmdArgs += @("--defense", $Defense) }
    if ($CaseId) { $cmdArgs += @("--case-id", $CaseId) }
    if ($LlmProvider) { $cmdArgs += @("--llm-provider", $LlmProvider) }
    if ($LlmModel) { $cmdArgs += @("--llm-model", $LlmModel) }
    if ($LlmBaseUrl) { $cmdArgs += @("--llm-base-url", $LlmBaseUrl) }
    if ($LlmApiKeysEnv) { $cmdArgs += @("--llm-api-keys-env", $LlmApiKeysEnv) }
    if ($Resume) { $cmdArgs += "--resume" }

    Write-Host "`n===== [$i/$total] $($c.id) -> $outDir =====" -ForegroundColor Cyan
    Write-Host "$Python $($cmdArgs -join ' ')" -ForegroundColor DarkGray
    & $Python @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "[$($c.id)] python exited with code $LASTEXITCODE (继续下一个 condition)"
    }
}

Write-Host "`n全部 condition 跑完。结果在 $Root\<id>\ 下。" -ForegroundColor Green
Write-Host "逐 condition 评估示例：python evaluate.py --results-dir $Root\C2 --out evals\baseline\C2\eval_C2.json --rule-only" -ForegroundColor Green
