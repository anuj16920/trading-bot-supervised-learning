# run_training.ps1 — Launch RL training and save all output to a timestamped log file
#
# Usage:
#   .\run_training.ps1                          # Phase 2 (3M steps)
#   .\run_training.ps1 -Phase3                  # Phase 3 (8M steps, domain randomization)
#   .\run_training.ps1 -Phase3 -Resume "checkpoints/rl/phase3/best/best_model.zip"
#   .\run_training.ps1 -Dummy                   # smoke-test with random data

param(
    [switch]$Phase3,
    [switch]$Dummy,
    [string]$Resume = "",
    [string]$Config = ""
)

# Build argument list
$args_list = @()
if ($Phase3)          { $args_list += "--phase3" }
if ($Dummy)           { $args_list += "--dummy" }
if ($Resume -ne "")   { $args_list += "--resume"; $args_list += $Resume }
if ($Config -ne "")   { $args_list += "--config"; $args_list += $Config }

# Create logs directory
$log_dir = "training_logs"
if (-not (Test-Path $log_dir)) { New-Item -ItemType Directory -Path $log_dir | Out-Null }

# Timestamped log filename
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$phase_tag = if ($Phase3) { "phase3" } else { "phase2" }
$log_file  = "$log_dir\train_${phase_tag}_${timestamp}.txt"

Write-Host "Logging to: $log_file" -ForegroundColor Cyan
Write-Host "Command: python train_rl.py $($args_list -join ' ')" -ForegroundColor Cyan
Write-Host ""

# Run training — tee sends output to both terminal and log file
python train_rl.py @args_list 2>&1 | Tee-Object -FilePath $log_file

Write-Host ""
Write-Host "Training complete. Log saved to: $log_file" -ForegroundColor Green
