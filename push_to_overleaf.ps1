# push_to_overleaf.ps1
# PowerShell script for Windows to sync local manuscript changes to Overleaf.
# Run this from the workspace root.

$ErrorActionPreference = "Stop"

# Get root directory of the workspace
$rootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $rootDir) {
    $rootDir = Get-Location
}

# Subfolder path where the LaTeX manuscript repository is located
$manuscriptSubfolder = Join-Path $rootDir "Journal-of-Real-Time-Image-Processing"

if (-not (Test-Path $manuscriptSubfolder)) {
    Write-Error "Error: Subfolder 'Journal-of-Real-Time-Image-Processing' not found at $manuscriptSubfolder"
    exit 1
}

# Change directory to the manuscript folder
Set-Location $manuscriptSubfolder
Write-Host "Working in manuscript directory: $manuscriptSubfolder" -ForegroundColor Yellow

# Verify that the 'overleaf' remote is configured
$remotes = git remote
if ($remotes -notcontains "overleaf") {
    Write-Host "Warning: The Git remote 'overleaf' is not configured in the manuscript repository." -ForegroundColor Yellow
    Write-Host "Please enter your Overleaf Git URL (e.g., https://git.overleaf.com/1234567890abcdef12345678):" -ForegroundColor Cyan
    $url = Read-Host "URL"
    if ($url) {
        $url = $url.Trim()
        git remote add overleaf $url
        Write-Host "Remote 'overleaf' added successfully!" -ForegroundColor Green
    } else {
        Write-Error "Error: Overleaf remote is required to sync."
        Set-Location $rootDir
        exit 1
    }
}

try {
    # 1. Fetch latest changes from Overleaf to ensure local tracking is fresh
    Write-Host "Fetching latest state from Overleaf..." -ForegroundColor Cyan
    git fetch overleaf

    # 2. Check out/reset overleaf-deploy branch to match Overleaf's remote master branch
    Write-Host "Recreating overleaf-deploy branch from overleaf/master..." -ForegroundColor Cyan
    git checkout -B overleaf-deploy overleaf/master

    # 3. Copy all tracked files from main branch
    Write-Host "Staging files from main branch..." -ForegroundColor Cyan
    git checkout main -- .

    # 4. Remove literature PDFs from the Overleaf branch
    if (Test-Path "literature") {
        Write-Host "Removing literature PDFs from Overleaf track..." -ForegroundColor Cyan
        git rm -r --cached literature/ 2>$null
        Remove-Item -Recurse -Force literature/
    }

    # 5. Commit changes if there are any
    $status = git status --porcelain
    if ($status) {
        Write-Host "Committing updates..." -ForegroundColor Cyan
        git commit -m "sync: update manuscript files to match main"
    } else {
        Write-Host "No new changes to commit." -ForegroundColor Green
    }

    # 6. Push to Overleaf
    Write-Host "Pushing to Overleaf master..." -ForegroundColor Cyan
    git push overleaf overleaf-deploy:master

    # 7. Return back to main branch
    Write-Host "Returning to main branch..." -ForegroundColor Cyan
    git checkout main

    Write-Host "Sync complete! Overleaf has been successfully updated with all your local changes." -ForegroundColor Green
}
catch {
    Write-Error "An error occurred during sync: $_"
    # Ensure we return to main branch on failure if we are on overleaf-deploy
    $currentBranch = (git branch --show-current).Trim()
    if ($currentBranch -eq "overleaf-deploy") {
        git checkout main
    }
}
finally {
    # Return to the root directory
    Set-Location $rootDir
}
