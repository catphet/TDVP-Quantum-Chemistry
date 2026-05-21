# run_all_validations.ps1
$scripts = @(
    "validate_n3.jl",
    "validate_n5.jl",
    "validate_n10.jl"#,
   # "validate_tang_n3.jl",
   # "validate_tang_n5.jl",
   # "validate_tang_n10.jl"
)

foreach ($script in $scripts) {
    if (Test-Path $script) {
        Write-Host "==============================" -ForegroundColor Cyan
        Write-Host "Running $script..." -ForegroundColor Cyan
        Write-Host "==============================" -ForegroundColor Cyan
        julia --project=.. $script
        Write-Host "Done: $script" -ForegroundColor Green
    } else {
        Write-Host "NOT FOUND: $script" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "All done! Compiling results..." -ForegroundColor Cyan
julia compile_results.jl