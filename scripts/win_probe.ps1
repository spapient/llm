# win_probe.ps1 - L0 read-only: locate big space consumers on C:
$ErrorActionPreference = "SilentlyContinue"
function GB($b) { if (-not $b) { $b = 0 }; [math]::Round($b/1GB, 2) }
function DirSize($path) {
  if (Test-Path $path) {
    (Get-ChildItem $path -Recurse -Force -File -ErrorAction SilentlyContinue |
      Measure-Object Length -Sum).Sum
  } else { 0 }
}

Write-Output "=== C盘根目录大文件 ==="
Get-ChildItem "C:\" -Force -File -ErrorAction SilentlyContinue |
  Sort-Object Length -Descending | Select-Object -First 10 |
  ForEach-Object { Write-Output ("{0} GB`t{1}" -f (GB $_.Length), $_.FullName) }

Write-Output "`n=== C盘一级目录占用 ==="
Get-ChildItem "C:\" -Force -Directory -ErrorAction SilentlyContinue | ForEach-Object {
  $s = DirSize $_.FullName
  Write-Output ("{0} GB`t{1}" -f (GB $s), $_.FullName)
}

Write-Output "`n=== 用户目录 $env:USERPROFILE 一级子目录 ==="
Get-ChildItem $env:USERPROFILE -Force -Directory -ErrorAction SilentlyContinue | ForEach-Object {
  $s = DirSize $_.FullName
  Write-Output ("{0} GB`t{1}" -f (GB $s), $_.FullName)
}

Write-Output "`n=== AppData 三大子目录 ==="
foreach ($d in @("$env:LOCALAPPDATA","$env:APPDATA","$env:USERPROFILE\AppData\LocalLow")) {
  Write-Output ("{0} GB`t{1}" -f (GB (DirSize $d)), $d)
}

Write-Output "`n=== AppData\Local 一级大目录 Top 15 ==="
Get-ChildItem "$env:LOCALAPPDATA" -Force -Directory -ErrorAction SilentlyContinue | ForEach-Object {
  [PSCustomObject]@{ GB = GB (DirSize $_.FullName); Path = $_.FullName }
} | Sort-Object GB -Descending | Select-Object -First 15 |
  ForEach-Object { Write-Output ("{0} GB`t{1}" -f $_.GB, $_.Path) }

Write-Output "`n=== ProgramData 一级大目录 Top 10 ==="
Get-ChildItem "C:\ProgramData" -Force -Directory -ErrorAction SilentlyContinue | ForEach-Object {
  [PSCustomObject]@{ GB = GB (DirSize $_.FullName); Path = $_.FullName }
} | Sort-Object GB -Descending | Select-Object -First 10 |
  ForEach-Object { Write-Output ("{0} GB`t{1}" -f $_.GB, $_.Path) }

Write-Output "`n=== 回收站(文件系统统计) ==="
$rb = DirSize "C:\`$Recycle.Bin"
Write-Output ("回收站: {0} GB" -f (GB $rb))

Write-Output "`n=== Windows.old ==="
if (Test-Path "C:\Windows.old") { Write-Output ("Windows.old: {0} GB" -f (GB (DirSize "C:\Windows.old"))) } else { Write-Output "不存在" }

Write-Output "`n=== 卷影副本/还原点占用 ==="
vssadmin list shadowstorage 2>$null | Select-String "Used|已使用|Allocated|分配" |
  ForEach-Object { Write-Output $_.Line.Trim() }

Write-Output "`n探测完成(未做任何修改)"
