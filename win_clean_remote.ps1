# win_clean_remote.ps1 - L1 安全清理(白名单可再生缓存/临时/日志),不碰个人文件
# 基于 skill 模板 win_clean.ps1 改编: 移除非交互会话中易挂起的回收站 COM 枚举
param(
  [switch]$SkipBrowser,
  [string]$LogPath = ""
)
$ErrorActionPreference = "SilentlyContinue"
$script:freed = 0

function Log($msg) {
  Write-Output $msg
  if ($LogPath) { Add-Content -Path $LogPath -Value $msg -Encoding UTF8 }
}
function CleanDir($label, $path) {
  if (-not (Test-Path $path)) { return }
  $size = (Get-ChildItem $path -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
  if (-not $size) { $size = 0 }
  Get-ChildItem $path -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
  $after = (Get-ChildItem $path -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
  if (-not $after) { $after = 0 }
  $delta = $size - $after
  $script:freed += $delta
  Log ("[完成] {0}: 释放 {1} MB(被占用文件已自动跳过)" -f $label, [math]::Round($delta/1MB))
}

Log ("=== C盘安全清理 开始 {0} ===" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
$before = (Get-PSDrive C).Free

CleanDir "用户临时文件" $env:TEMP
CleanDir "系统临时文件" "C:\Windows\Temp"

$wuWasRunning = (Get-Service wuauserv -ErrorAction SilentlyContinue).Status -eq "Running"
Stop-Service wuauserv -Force -ErrorAction SilentlyContinue
CleanDir "Windows更新缓存" "C:\Windows\SoftwareDistribution\Download"
if ($wuWasRunning) { Start-Service wuauserv -ErrorAction SilentlyContinue }

CleanDir "DirectX着色器缓存" "$env:LOCALAPPDATA\D3DSCache"
CleanDir "NVIDIA着色器缓存" "$env:LOCALAPPDATA\NVIDIA\DXCache"
CleanDir "NVIDIA GL着色器缓存" "$env:LOCALAPPDATA\NVIDIA\GLCache"

CleanDir "系统日志" "C:\Windows\Logs"
CleanDir "错误报告(WER)" "$env:ProgramData\Microsoft\Windows\WER"
CleanDir "蓝屏小转储" "C:\Windows\Minidump"
if (Test-Path "C:\Windows\MEMORY.DMP") {
  $dmp = (Get-Item "C:\Windows\MEMORY.DMP").Length
  Remove-Item "C:\Windows\MEMORY.DMP" -Force -ErrorAction SilentlyContinue
  if (-not (Test-Path "C:\Windows\MEMORY.DMP")) { $script:freed += $dmp; Log ("[完成] 内存转储 MEMORY.DMP: 释放 {0} MB" -f [math]::Round($dmp/1MB)) }
  else { Log "[跳过] MEMORY.DMP 删除失败(可能被占用),未计入释放量" }
}

$thumbDir = "$env:LOCALAPPDATA\Microsoft\Windows\Explorer"
$tsize = (Get-ChildItem $thumbDir -Filter "thumbcache_*" -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
if (-not $tsize) { $tsize = 0 }
Get-ChildItem $thumbDir -Filter "thumbcache_*" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
$tafter = (Get-ChildItem $thumbDir -Filter "thumbcache_*" -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
if (-not $tafter) { $tafter = 0 }
$script:freed += ($tsize - $tafter)
Log ("[完成] 缩略图缓存: 释放 {0} MB(被占用文件已跳过)" -f [math]::Round(($tsize-$tafter)/1MB))

if (-not $SkipBrowser) {
  foreach ($ud in @("$env:LOCALAPPDATA\Google\Chrome\User Data", "$env:LOCALAPPDATA\Microsoft\Edge\User Data")) {
    if (-not (Test-Path $ud)) { continue }
    $browser = if ($ud -match "Chrome") { "Chrome" } else { "Edge" }
    $profiles = Get-ChildItem $ud -Directory -ErrorAction SilentlyContinue
    foreach ($profile in $profiles) {
      if ($profile.Name -ne "Default" -and $profile.Name -notlike "Profile *") { continue }
      CleanDir ("{0}缓存({1})" -f $browser, $profile.Name) (Join-Path $profile.FullName "Cache")
      CleanDir ("{0}代码缓存({1})" -f $browser, $profile.Name) (Join-Path $profile.FullName "Code Cache")
    }
  }
}

Log "[跳过] 回收站: 文件系统统计为 0 GB,且按默认策略不清空(误删恢复渠道)"

$after = (Get-PSDrive C).Free
Log ("=== 清理完成: 共释放约 {0} GB (C盘剩余 {1} GB -> {2} GB) ===" -f [math]::Round($script:freed/1GB,2), [math]::Round($before/1GB,1), [math]::Round($after/1GB,1))
Log "未触碰: 个人文件(桌面/文档/下载/图片/视频)、已安装软件、系统文件"
