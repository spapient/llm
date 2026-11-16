# win_delete_old.ps1 - 删除经核实的旧 llmu 验证/构建目录(白名单,逐个校验父目录)
# 默认 DryRun 只统计; 加 -Execute 才实际删除
param(
  [switch]$Execute,
  [string]$LogPath = "C:\Users\fei\llmu_cleanup_log.txt"
)
$ErrorActionPreference = "SilentlyContinue"

$delete = @(
  # 非 git 构建/验证产物
  "llm-sycl-verify-tree",
  "llmu-cfgspec",
  "llmu-cudamerge",
  "llmu-cur",
  "llmu-di",
  "llmu-ffi1",
  "llmu-ffi2",
  "llmu-fp16kv",
  "llmu-fp16kv2",
  "llmu-verify-20261022",
  "llmu-verify-20261023",
  "llmu-verify-20261024",
  "llmu-verify-20261029",
  "llmu-verify-20261105",
  "llmu-verify-config",
  "llmu-wip-20261030-0224",
  "llmu-x86-base-bd1d94f",
  "llmu-x86-new-avx5",
  # 干净 git 克隆(历史已由本地 .bundle 或 GitHub 保留)
  "llm-sycl-verify",
  "llmu-verify-20261026",
  "llmu-verify-20261027-0a80bc9",
  "llmu-verify-20261027-1f91658",
  "llmu-verify-20261027-2ace644",
  "llmu-verify-20261027-383a359",
  "llmu-verify-20261027-4839a71",
  "llmu-verify-20261027-6a8ed4b",
  "llmu-verify-20261027-8bfa248",
  "llmu-verify-20261027-9095bb4",
  "llmu-verify-20261027-96b66c4",
  "llmu-verify-20261027-ac23057",
  "llmu-verify-20261027-b69166a",
  "llmu-verify-20261027-be296d2",
  "llmu-verify-20261027-e9e1b34",
  "verify-llm"
)

$homeDir = $env:USERPROFILE
$freed = 0
$skipped = 0

function Log($msg) {
  Write-Output $msg
  if ($LogPath) { Add-Content -Path $LogPath -Value $msg -Encoding UTF8 }
}

Log ("=== 旧构建目录清理 {0} {1} ===" -f $(if ($Execute) { "执行" } else { "预演" }), (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
$before = (Get-PSDrive C).Free

foreach ($name in $delete) {
  $path = Join-Path $homeDir $name
  # 安全校验: 必须位于用户主目录下,且不是符号链接/联接
  $parent = Split-Path $path -Parent
  if ($parent -ne $homeDir) { Log "[跳过] 路径越界: $path"; $skipped++; continue }
  if (-not (Test-Path $path)) { Log "[不存在] $name"; continue }
  $item = Get-Item $path -Force
  if ($item.LinkType) { Log "[跳过] 联接/符号链接,未处理: $name ($($item.LinkType))"; $skipped++; continue }

  $size = (Get-ChildItem $path -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
  if (-not $size) { $size = 0 }

  if (-not $Execute) {
    Log ("[预演] {0}: {1} GB" -f $name, [math]::Round($size/1GB,2))
    continue
  }

  Remove-Item $path -Recurse -Force -ErrorAction SilentlyContinue
  if (Test-Path $path) {
    Log "[部分失败] $name 仍有残留(可能被占用)"
  } else {
    $freed += $size
    Log ("[已删除] {0}: {1} GB" -f $name, [math]::Round($size/1GB,2))
  }
}

$after = (Get-PSDrive C).Free
if ($Execute) {
  Log ("=== 完成: 脚本计释放 {0} GB; C盘剩余 {1} GB -> {2} GB; 跳过 {3} 项 ===" -f `
    [math]::Round($freed/1GB,2), [math]::Round($before/1GB,1), [math]::Round($after/1GB,1), $skipped)
} else {
  Log "=== 预演结束,未删除任何文件 ==="
}
