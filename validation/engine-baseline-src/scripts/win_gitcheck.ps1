# win_gitcheck.ps1 - L0: 检查各 git 目录的远程与未推送提交
$ErrorActionPreference = "SilentlyContinue"
$pat = "^(llmu-verify|llm-sycl-verify|llmu-wip|llmwork|llmu-cfgspec|llmu-cudamerge|llmu-ffi|llmu-fp16|llmu-x86|verify-llm|llmu-cur|llmu-di)"

$rows = Get-ChildItem $env:USERPROFILE -Force -Directory -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match $pat -and (Test-Path (Join-Path $_.FullName ".git")) }

foreach ($d in $rows) {
  $remote = (& git -C $d.FullName remote get-url origin 2>$null)
  $branch = (& git -C $d.FullName branch --show-current 2>$null)
  $up = (& git -C $d.FullName rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>$null)
  $ahead = 0
  if ($up) { $ahead = (& git -C $d.FullName rev-list --count '@{u}..HEAD' 2>$null) }
  $dirty = (& git -C $d.FullName status --porcelain 2>$null | Measure-Object).Count
  if (-not $up) { $up = "无" }
  if (-not $remote) { $remote = "无远程" }
  Write-Output ("{0}`tDirty={1}`tBranch={2}`tUpstream={3}`tAhead={4}`tRemote={5}" -f `
    $d.Name, $dirty, $branch, $up, $ahead, $remote)
}
