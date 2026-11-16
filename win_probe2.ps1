# win_probe2.ps1 - L0 read-only: second-pass breakdown
$ErrorActionPreference = "SilentlyContinue"
function GB($b) { if (-not $b) { $b = 0 }; [math]::Round($b/1GB, 2) }
function DirSize($path) {
  if (Test-Path $path) {
    (Get-ChildItem $path -Recurse -Force -File -ErrorAction SilentlyContinue |
      Measure-Object Length -Sum).Sum
  } else { 0 }
}

Write-Output "=== 根目录系统文件(隐藏) ==="
Get-ChildItem "C:\" -Force -File -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match "pagefile|hiberfil|swapfile|DumpStack" } |
  ForEach-Object { Write-Output ("{0} GB`t{1}`t{2}" -f (GB $_.Length), $_.Name, $_.LastWriteTime) }

Write-Output "`n=== Roaming 一级大目录 Top 15 ==="
Get-ChildItem "$env:APPDATA" -Force -Directory -ErrorAction SilentlyContinue | ForEach-Object {
  [PSCustomObject]@{ GB = GB (DirSize $_.FullName); Path = $_.FullName }
} | Sort-Object GB -Descending | Select-Object -First 15 |
  ForEach-Object { Write-Output ("{0} GB`t{1}" -f $_.GB, $_.Path) }

Write-Output "`n=== .cache 一级子目录 ==="
Get-ChildItem "$env:USERPROFILE\.cache" -Force -Directory -ErrorAction SilentlyContinue | ForEach-Object {
  [PSCustomObject]@{ GB = GB (DirSize $_.FullName); Path = $_.FullName }
} | Sort-Object GB -Descending |
  ForEach-Object { Write-Output ("{0} GB`t{1}" -f $_.GB, $_.Path) }

Write-Output "`n=== WSL 磁盘文件 ==="
Get-ChildItem "$env:LOCALAPPDATA\wsl" -Recurse -Force -File -ErrorAction SilentlyContinue |
  Where-Object { $_.Extension -in ".vhdx",".vhd" } |
  ForEach-Object { Write-Output ("{0} GB`t{1}`t{2}" -f (GB $_.Length), $_.FullName, $_.LastWriteTime) }
Get-ChildItem "C:\WSL" -Recurse -Force -File -ErrorAction SilentlyContinue |
  Where-Object { $_.Extension -in ".vhdx",".vhd" } |
  ForEach-Object { Write-Output ("{0} GB`t{1}`t{2}" -f (GB $_.Length), $_.FullName, $_.LastWriteTime) }

Write-Output "`n=== 旧验证/构建目录(只报告,按修改时间) ==="
Get-ChildItem "$env:USERPROFILE" -Force -Directory -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match "llmu-verify|llm-sycl|llmu-wip|llmwork|llmu-cfgspec|llmu-cudamerge|llmu-ffi|llmu-fp16|llmu-x86|verify-llm|llmu-cur|llmu-di" } |
  ForEach-Object {
    [PSCustomObject]@{ GB = GB (DirSize $_.FullName); Last = $_.LastWriteTime.ToString("yyyy-MM-dd"); Path = $_.FullName }
  } | Sort-Object GB -Descending |
  ForEach-Object { Write-Output ("{0} GB`t{1}`t{2}" -f $_.GB, $_.Last, $_.Path) }

Write-Output "`n=== Downloads 大文件 Top 10 ==="
Get-ChildItem "$env:USERPROFILE\Downloads" -Recurse -Force -File -ErrorAction SilentlyContinue |
  Sort-Object Length -Descending | Select-Object -First 10 |
  ForEach-Object { Write-Output ("{0} GB`t{1}`t{2}" -f (GB $_.Length), $_.LastWriteTime.ToString("yyyy-MM-dd"), $_.FullName) }

Write-Output "`n=== 休眠状态 ==="
powercfg /a 2>$null | Select-Object -First 6 | ForEach-Object { Write-Output $_.Trim() }

Write-Output "`n探测2完成(未做任何修改)"
