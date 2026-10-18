<#
.SYNOPSIS
  Evaluate every experiment condition under a result root.

.DESCRIPTION
  Each condition directory containing JSON traces is evaluated independently.
  Summaries are written to <EvalRoot>\<run-family>\<condition>\.

.PARAMETER Root
  Result root produced by run_matrix.ps1, for example results\raw.

.PARAMETER EvalRoot
  Root for evaluation summaries. Default: evals.

.PARAMETER Dataset
  Ground-truth dataset. Default: dataset\pilot_tasks_mas.json.

.PARAMETER Workers
  Number of traces judged concurrently within each condition.

.PARAMETER FolderParallel
  Evaluate condition directories in parallel PowerShell jobs. This multiplies
  peak judge concurrency by the number of simultaneous folders.

.PARAMETER RuleOnly
  Run deterministic evaluation without an LLM judge.

.PARAMETER CostOnly
  Report token and defense-gate costs without an LLM judge.

.PARAMETER JudgeBackend
  LLM judge backend: native or mas-eval.

.PARAMETER MemoryBank
  MAS-Eval memory-bank path. Required with JudgeBackend=mas-eval unless RuleOnly
  or CostOnly is selected.

.PARAMETER Python
  Python executable or command available to PowerShell. Default: python.

.EXAMPLE
  .\scripts\eval_matrix.ps1 -Root results\raw -RuleOnly

.EXAMPLE
  .\scripts\eval_matrix.ps1 -Root results\source -Workers 8 -FolderParallel
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 2147483647)][int]$Workers = 8,
    [string]$Root = "results\raw",
    [string]$EvalRoot = "evals",
    [string]$Dataset = "dataset\pilot_tasks_mas.json",
    [string]$Python = "python",
    [switch]$RuleOnly,
    [switch]$CostOnly,
    [string[]]$Only,
    [switch]$FolderParallel,
    [ValidateSet("native", "mas-eval")][string]$JudgeBackend = "native",
    [string]$MemoryBank = "",
    [string]$OutputSuffix = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (-not $env:PYTHONIOENCODING) {
    $env:PYTHONIOENCODING = "utf-8"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repoRoot

if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
    throw "results root not found: $Root"
}
if (-not (Test-Path -LiteralPath $Dataset -PathType Leaf)) {
    throw "dataset not found: $Dataset"
}
if (-not $RuleOnly -and -not $CostOnly -and
    $JudgeBackend -eq "mas-eval" -and -not $MemoryBank) {
    throw "JudgeBackend=mas-eval requires -MemoryBank"
}
if ($MemoryBank -and -not (Test-Path -LiteralPath $MemoryBank -PathType Leaf)) {
    throw "memory bank not found: $MemoryBank"
}
if (-not $OutputSuffix) {
    $OutputSuffix = if ($JudgeBackend -eq "mas-eval") { "_mas_eval" } else { "" }
}

$dirs = @(
    Get-ChildItem -LiteralPath $Root -Directory | Where-Object {
        Get-ChildItem -LiteralPath $_.FullName -Filter *.json -File |
            Select-Object -First 1
    }
)
if ($Only) {
    $dirs = @($dirs | Where-Object { $Only -contains $_.Name })
}
if (-not $dirs.Count) {
    throw "no condition subfolders with JSON traces under $Root"
}

function Get-EvalArguments {
    param([System.IO.DirectoryInfo]$Directory)

    $resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
    $runFamily = Split-Path -Leaf $resolvedRoot
    $outDir = Join-Path (Join-Path $EvalRoot $runFamily) $Directory.Name
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $outFile = Join-Path $outDir (
        "eval_{0}{1}.json" -f $Directory.Name, $OutputSuffix
    )
    $arguments = @(
        "evaluate.py",
        "--results-dir", $Directory.FullName,
        "--dataset", $Dataset,
        "--out", $outFile,
        "--workers", [string]$Workers
    )
    if ($RuleOnly) {
        $arguments += "--rule-only"
    }
    elseif ($CostOnly) {
        $arguments += "--cost-only"
    }
    elseif ($JudgeBackend -ne "native") {
        $arguments += @("--judge-backend", $JudgeBackend)
        if ($MemoryBank) {
            $arguments += @("--memory-bank", $MemoryBank)
        }
    }
    return ,$arguments
}

Write-Host (
    "Evaluating {0} condition(s): {1}" -f
    $dirs.Count, ($dirs.Name -join ", ")
) -ForegroundColor Cyan

if ($FolderParallel) {
    $jobs = @(
        foreach ($directory in $dirs) {
            $arguments = Get-EvalArguments $directory
            Start-Job -Name $directory.Name `
                -ArgumentList $repoRoot, $Python, $arguments `
                -ScriptBlock {
                    param($WorkingDirectory, $PythonCommand, $Arguments)
                    Set-Location -LiteralPath $WorkingDirectory
                    & $PythonCommand @Arguments
                    if ($LASTEXITCODE -ne 0) {
                        throw "evaluate.py exited with code $LASTEXITCODE"
                    }
                }
        }
    )
    $jobs | Wait-Job | Out-Null
    foreach ($job in $jobs) {
        Write-Host "`n===== [$($job.Name)] =====" -ForegroundColor Cyan
        Receive-Job $job
        if ($job.State -ne "Completed") {
            throw "evaluation job $($job.Name) ended in state $($job.State)"
        }
    }
    $jobs | Remove-Job
}
else {
    $index = 0
    foreach ($directory in $dirs) {
        $index++
        $arguments = Get-EvalArguments $directory
        Write-Host (
            "`n===== [{0}/{1}] {2} =====" -f
            $index, $dirs.Count, $directory.Name
        ) -ForegroundColor Cyan
        & $Python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "evaluate.py failed for $($directory.Name): exit $LASTEXITCODE"
        }
    }
}

Write-Host "Evaluation complete. Summaries are under $EvalRoot." -ForegroundColor Green
