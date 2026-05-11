param(
    [ValidateSet("val2017", "train2017")]
    [string]$Split = "val2017"
)

$ErrorActionPreference = "Stop"

$urls = @{
    "val2017" = "http://images.cocodataset.org/zips/val2017.zip"
    "train2017" = "http://images.cocodataset.org/zips/train2017.zip"
}

$outDir = "data\train\content"
$zipPath = Join-Path $outDir "$Split.zip"
$extractDir = Join-Path $outDir $Split

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Write-Output "Downloading $Split from $($urls[$Split])"
curl.exe -L --retry 10 --retry-delay 5 --continue-at - --output $zipPath $urls[$Split]
if ($LASTEXITCODE -ne 0) {
    throw "curl failed with exit code $LASTEXITCODE"
}

Write-Output "Extracting $zipPath"
Expand-Archive -Path $zipPath -DestinationPath $outDir -Force

Write-Output "Done: $extractDir"
