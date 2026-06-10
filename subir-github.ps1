$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$remoteUrl = "https://github.com/PedroSilvazDev/trabalho-final-ia-diabetes-colab.git"
$commitMessage = "Trabalho final IA - versao Google Colab"

if (-not (Test-Path ".\.git")) {
    git init
    git branch -M main
}

$remotes = git remote 2>$null
if ($remotes -contains "origin") {
    git remote set-url origin $remoteUrl
} else {
    git remote add origin $remoteUrl
}

git add .
git commit -m $commitMessage 2>$null
if ($LASTEXITCODE -ne 0) {
    git commit -m $commitMessage
}

git push -u origin main

Write-Host ""
Write-Host "Repositorio: https://github.com/PedroSilvazDev/trabalho-final-ia-diabetes-colab" -ForegroundColor Green
Write-Host "Colab: https://colab.research.google.com/github/PedroSilvazDev/trabalho-final-ia-diabetes-colab/blob/main/trabalho_final_ia_colab.ipynb" -ForegroundColor Green
