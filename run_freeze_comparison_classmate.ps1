$ErrorActionPreference = "Stop"
Set-Location "E:\plant_disease\egwt-reproduction"

$epochs = 40
$ckpt = "checkpoints/imagenet_pretrain_classmate/egwt_imagenet_best.pt"
$datasets = @("plantvillage", "cassava", "tomato")
$modes = @("none", "stage12", "head_only")

foreach ($mode in $modes) {
    foreach ($ds in $datasets) {
        Write-Host "===== [classmate pretrain] freeze_mode=$mode dataset=$ds =====" -ForegroundColor Cyan
        python -u train.py --data_dir "data/$ds" --dataset_name $ds --epochs $epochs --freeze_mode $mode --imagenet_ckpt $ckpt --out_dir runs_realpretrain_classmate
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED: freeze_mode=$mode dataset=$ds (exit $LASTEXITCODE)" -ForegroundColor Red
        }
    }
}

Write-Host "===== ALL CLASSMATE-PRETRAIN FREEZE-COMPARISON RUNS DONE =====" -ForegroundColor Green
