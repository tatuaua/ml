# Run all tests and main.go from this script's directory.
Push-Location $PSScriptRoot
try {
    Write-Host "Running tests..." -ForegroundColor Cyan
    go test ./...
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Tests failed!" -ForegroundColor Red
        exit 1
    }

    Write-Host "`nRunning main.go..." -ForegroundColor Cyan
    go run .
}
finally {
    Pop-Location
}
