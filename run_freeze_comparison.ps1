$ErrorActionPreference = "Stop"
Set-Location "E:\plant_disease\egwt-reproduction"

$epochs = 40
$datasets = @("plantvillage", "cassava", "tomato")
$modes = @("stage12", "head_only")

foreach ($mode in $modes) {
    foreach ($ds in $datasets) {
        Write-Host "===== freeze_mode=$mode dataset=$ds =====" -ForegroundColor Cyan
        python -u train.py --data_dir "data/$ds" --dataset_name $ds --epochs $epochs --freeze_mode $mode --out_dir runs
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED: freeze_mode=$mode dataset=$ds (exit $LASTEXITCODE)" -ForegroundColor Red
        }
    }
}

Write-Host "===== ALL FREEZE-COMPARISON RUNS DONE =====" -ForegroundColor Green
