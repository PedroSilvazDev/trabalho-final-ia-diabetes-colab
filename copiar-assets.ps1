$ErrorActionPreference = "Stop"

$figuresSrc = "C:\Users\phsil\Projects\trabalho-final-ia-diabetes\outputs\figures"
$figuresDst = "$PSScriptRoot\outputs\figures"
$pdfSrc = "C:\Users\phsil\Downloads\relatorio_trabalho_ia.pdf"
$pdfDst = "$PSScriptRoot\docs\relatorio.pdf"

New-Item -ItemType Directory -Path $figuresDst -Force | Out-Null
New-Item -ItemType Directory -Path "$PSScriptRoot\docs" -Force | Out-Null

if (Test-Path $figuresSrc) {
    Copy-Item "$figuresSrc\*.png" $figuresDst -Force
    Write-Host "Graficos copiados para outputs\figures\"
} else {
    Write-Host "Aviso: pasta de graficos nao encontrada. Rode python main.py no projeto principal."
}

if (Test-Path $pdfSrc) {
    Copy-Item $pdfSrc $pdfDst -Force
    Write-Host "PDF copiado para docs\relatorio.pdf"
} else {
    Write-Host "Aviso: PDF nao encontrado em Downloads."
}
