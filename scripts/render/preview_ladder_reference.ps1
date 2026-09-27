param(
    [string]$Python = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ViewerArgs = @()
)

$ErrorActionPreference = "Stop"
if (-not $Python) {
    $PreviewPython = Join-Path $env:USERPROFILE "anaconda3\envs\teleopit\python.exe"
    if (Test-Path -LiteralPath $PreviewPython) {
        $Python = $PreviewPython
    } else {
        $Python = (Get-Command python -ErrorAction Stop).Source
    }
}
$PreviewScript = Join-Path $PSScriptRoot "preview_ladder_reference.py"
& $Python $PreviewScript @ViewerArgs
exit $LASTEXITCODE
