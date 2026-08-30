$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$historyPath = Join-Path $projectRoot "results\inaturalist\training_history.json"
$statusPath = Join-Path $projectRoot "shutdown_monitor_status.txt"
$trainingProcessIds = @(26592, 20068)

Set-Content -LiteralPath $statusPath -Encoding UTF8 -Value (
    "{0:yyyy-MM-dd HH:mm:ss} Monitoring iNaturalist epoch 11." -f (Get-Date)
)

while ($true) {
    if (Test-Path -LiteralPath $historyPath) {
        try {
            $history = Get-Content -Raw -LiteralPath $historyPath | ConvertFrom-Json
            $lastEpoch = [int](($history.history | Select-Object -Last 1).epoch)
            if ($lastEpoch -ge 11) {
                Add-Content -LiteralPath $statusPath -Encoding UTF8 -Value (
                    "{0:yyyy-MM-dd HH:mm:ss} Epoch {1} completed. Shutdown requested." -f (
                        Get-Date
                    ), $lastEpoch
                )
                & "$env:SystemRoot\System32\shutdown.exe" /s /t 30 /c (
                    "iNaturalist epoch 11 completed. Automatic shutdown."
                )
                exit 0
            }
        }
        catch {
            Add-Content -LiteralPath $statusPath -Encoding UTF8 -Value (
                "{0:yyyy-MM-dd HH:mm:ss} Waiting for a complete history update." -f (
                    Get-Date
                )
            )
        }
    }

    $trainingIsRunning = $false
    foreach ($processId in $trainingProcessIds) {
        if (Get-Process -Id $processId -ErrorAction SilentlyContinue) {
            $trainingIsRunning = $true
            break
        }
    }
    if (-not $trainingIsRunning) {
        Add-Content -LiteralPath $statusPath -Encoding UTF8 -Value (
            "{0:yyyy-MM-dd HH:mm:ss} Training stopped before epoch 11; no shutdown." -f (
                Get-Date
            )
        )
        exit 2
    }

    Start-Sleep -Seconds 20
}
