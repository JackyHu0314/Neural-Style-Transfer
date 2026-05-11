$ErrorActionPreference = "Stop"

Write-Output "This script uses the Hugging Face CLI. Install it first if needed:"
Write-Output "pip install huggingface_hub"
Write-Output ""
Write-Output "Then run:"
Write-Output "huggingface-cli download Dant33/WikiArt-81K-BLIP_2-1024x1024 --repo-type dataset --local-dir data\train\style\wikiart"

