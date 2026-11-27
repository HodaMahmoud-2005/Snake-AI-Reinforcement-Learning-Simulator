param(
    [string]$ProjectRoot = (Get-Location).Path
)

$ErrorActionPreference = 'Stop'
$root = [System.IO.Path]::GetFullPath($ProjectRoot)
if (-not (Test-Path -LiteralPath $root -PathType Container)) {
    throw "Project directory does not exist: $root"
}
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Add-NewProjectFile {
    param([string]$RelativePath, [string]$Content)
    $path = Join-Path $root $RelativePath
    if (Test-Path -LiteralPath $path) {
        Write-Host "SKIPPED (already exists): $RelativePath"
        return
    }
    [System.IO.File]::WriteAllText($path, $Content, $utf8)
    Write-Host "CREATED: $RelativePath" -ForegroundColor Green
}

Write-Host "Creating project structure in: $root"
foreach ($folder in @('models', 'results', 'tests', 'report')) {
    $path = Join-Path $root $folder
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        New-Item -ItemType Directory -Path $path | Out-Null
        Write-Host "CREATED: $folder/" -ForegroundColor Green
    }
}

$modules = [ordered]@{
    'environment.py' = '# TODO (Hany, #03-04): Implement the real Snake environment and deterministic transitions.'
    'value_iteration.py' = '# TODO (Hoda, #06-09): BFS, transition model, value iteration and policy extraction.'
    'q_learning.py' = '# TODO (Basmala #11, Jowairya #12): Q-table updates and epsilon-greedy exploration.'
    'reinforce.py' = '# TODO (Mohamed #16, Jowairya #17): Softmax policy and REINFORCE updates.'
    'train.py' = '# TODO (Basmala, #13-14): Shared training/planning pipeline and model persistence.'
    'evaluate.py' = '# TODO (Jowairya, #22): Shared evaluation, metrics and CSV exports.'
    'game_ui.py' = '# TODO (Mohamed, #21): Pygame interface and trained-agent demos.'
    'plots.py' = '# TODO (Mohamed, #18-19): Learning curves and comparison charts.'
    'main.py' = '# TODO (Basmala, #23): Train, evaluate and demo application modes.'
    'config.py' = '# TODO (Hany, #02): Shared game settings and hyperparameters.'
}
foreach ($name in $modules.Keys) {
    Add-NewProjectFile $name ($modules[$name] + "`n")
}

Add-NewProjectFile 'requirements.txt' "numpy`npygame`nmatplotlib`npandas`npytest`n"
Add-NewProjectFile '.gitignore' @'
# Python caches and coverage
__pycache__/
*.py[cod]
.pytest_cache/
.coverage
htmlcov/

# Local environments and secrets
.venv/
venv/
env/
.env
.env.*
!.env.example

# Local temporary files
*.tmp
*.swp
.DS_Store
Thumbs.db
'@
Add-NewProjectFile 'README.md' "# Snake AI - Reinforcement Learning Simulator`n`nProject documentation in progress.`n"
foreach ($folder in @('models', 'results', 'report')) {
    Add-NewProjectFile "$folder/.gitkeep" ''
}

# The first five files are explicitly named in the issue backlog.
# The rest are suggested dedicated tests for existing acceptance criteria.
$tests = [ordered]@{
    'test_environment.py' = @'
# Owner: Hany | Backlog #05
# TODO: Test reset, relative turns, movement, growth, food placement and tail-cell movement.
# TODO: Test wall/obstacle/body collisions, rewards, win and step-limit truncation.
# TODO: Test deterministic transition purity and step/transition consistency.
'@
    'test_value_iteration.py' = @'
# Owner: Hoda | Backlog #10
# TODO: Test reachable state coverage, Bellman update and terminal transitions.
# TODO: Test valid policy actions, convergence residual and known real-game scenarios.
'@
    'test_q_learning.py' = @'
# Owner: Basmala | Backlog #15 (coordinates with Jowairya #12)
# TODO: Test ordinary and terminal TD updates and truncation handling.
# TODO: Test epsilon extremes, decay floor, seeded actions and unseen states.
'@
    'test_reinforce.py' = @'
# Owner: Mohamed | Backlog #20 (coordinates with Jowairya #17)
# TODO: Test Softmax normalization and numerical stability.
# TODO: Test discounted returns, gradient direction and seeded reproducibility.
'@
    'test_integration.py' = @'
# Owner: Basmala | Backlog #23
# TODO: Test train/evaluate/demo modes using real implemented modules.
# TODO: Test invalid arguments, missing/incompatible models and headless evaluation.
# TODO: Run end-to-end smoke tests for all three algorithms.
'@
    'test_train.py' = @'
# Suggested owner: Basmala | Backlog #13
# TODO: Test algorithm selection, small training/planning budgets and log fields.
# TODO: Check seeded reproducibility and separate DP convergence logs.
'@
    'test_model_io.py' = @'
# Suggested owner: Basmala | Backlog #14
# TODO: Test save/load round trips for policy, Q-table and logits with metadata.
# TODO: Check full-state keys, incompatible configuration and action consistency.
'@
    'test_evaluate.py' = @'
# Suggested owner: Jowairya | Backlog #22
# TODO: Test metrics, no learning during evaluation and CSV output.
# TODO: Check shared evaluation schedules, seeds and handling of no successful episodes.
'@
    'test_game_ui.py' = @'
# Suggested owner: Mohamed | Backlog #21
# TODO: Test real environment rendering, start/reset and algorithm selection.
# TODO: Smoke-test trained-agent demos; use a headless display where supported.
'@
    'test_plots.py' = @'
# Suggested owner: Mohamed | Backlog #18-19
# TODO: Test output files, labeled plots, metric columns and missing winning episodes.
# TODO: Use actual training/evaluation records for final comparison charts.
'@
}
foreach ($name in $tests.Keys) {
    Add-NewProjectFile "tests/$name" ($tests[$name] + "`n")
}

Write-Host "`nProject scaffold ready. Existing files were preserved." -ForegroundColor Cyan
Write-Host 'Test files contain TODO checklists only; no tests have been implemented yet.'
Write-Host 'Next: python -m pip install -r requirements.txt'
Write-Host 'After implementing tests: python -m pytest -v'
