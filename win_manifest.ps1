# win_manifest.ps1 - L0: 核实待删旧构建目录清单(大小/时间/git 状态),不删除
$ErrorActionPreference = "SilentlyContinue"
function GB($b) { if (-not $b) { $b = 0 }; [math]::Round($b/1GB, 2) }
function DirSize($path) {
  (Get-ChildItem $path -Recurse -Force -File -ErrorAction SilentlyContinue |
    Measure-Object Length -Sum).Sum
}
$pat = "^(llmu-verify|llm-sycl-verify|llmu-wip|llmwork|llmu-cfgspec|llmu-cudamerge|llmu-ffi|llmu-fp16|llmu-x86|verify-llm|llmu-cur|llmu-di)"

$rows = Get-ChildItem $env:USERPROFILE -Force -Directory -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match $pat }

Write-Output ("匹配目录数: {0}" -f $rows.Count)
$total = 0
foreach ($d in $rows) {
  $size = DirSize $d.FullName
  $total += $size
  $git = Test-Path (Join-Path $d.FullName ".git")
  $dirty = ""; $head = ""
  if ($git) {
    $dirty = (& git -C $d.FullName status --porcelain 2>$null | Measure-Object).Count
    $head = (& git -C $d.FullName rev-parse --short HEAD 2>$null)
  }
  Write-Output ("{0} GB`t{1}`tGit={2}`tDirty={3}`tHead={4}`t{5}" -f `
    (GB $size), $d.LastWriteTime.ToString("yyyy-MM-dd HH:mm"), $git, $dirty, $head, $d.Name)
}
Write-Output ("合计: {0} GB" -f (GB $total))
