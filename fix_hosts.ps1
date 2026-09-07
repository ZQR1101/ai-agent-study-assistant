$hostsPath = "$env:SystemRoot\System32\drivers\etc\hosts"
$content = Get-Content $hostsPath -Raw
$newContent = $content -replace "(?m)^127\.0\.0\.1\s+huggingface\.co\s*$", ""
Set-Content -Path $hostsPath -Value $newContent -Encoding UTF8 -Force
Write-Host "Done. Removed huggingface.co entries."
