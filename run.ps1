# Run all tests and main.go
Write-Host "Running tests..." -ForegroundColor Cyan
go test ./...
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tests failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`nRunning main.go..." -ForegroundColor Cyan
go run .
