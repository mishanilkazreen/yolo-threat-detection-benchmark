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
    Write-Host "Overleaf now requires Git authentication tokens." -ForegroundColor LightCyan
    Write-Host "Please generate a Git token in your Overleaf Account Settings and enter the URL in this format:" -ForegroundColor LightCyan
    Write-Host "  https://git:<your-token>@git.overleaf.com/<project-id>" -ForegroundColor LightCyan
    Write-Host "Please enter your Overleaf Git URL:" -ForegroundColor Cyan
    $url = Read-Host "URL"
    if ($url) {
        $url = $url.Trim()
        git remote add overleaf $url
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to add git remote."
            Set-Location $rootDir
            exit 1
        }
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
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to fetch from Overleaf. This is usually due to incorrect credentials or Git tokens. See: https://www.overleaf.com/learn/how-to/Git_integration_authentication_tokens"
    }

    # 2. Check out/reset overleaf-deploy branch to match Overleaf's remote master branch
    Write-Host "Recreating overleaf-deploy branch from overleaf/master..." -ForegroundColor Cyan
    git checkout -B overleaf-deploy overleaf/master
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to checkout overleaf-deploy branch from overleaf/master. Check if Overleaf master branch exists."
    }

    # 3. Copy all tracked files from main branch
    Write-Host "Staging files from main branch..." -ForegroundColor Cyan
    git checkout main -- .
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to copy files from main branch to overleaf-deploy branch."
    }

    # 4. Remove literature PDFs from the Overleaf branch
    if (Test-Path "literature") {
        Write-Host "Removing literature PDFs from Overleaf track..." -ForegroundColor Cyan
        git rm -r --cached literature/ 2>$null
        if ($LASTEXITCODE -ne 0) {
            $global:LASTEXITCODE = 0
        }
        Remove-Item -Recurse -Force literature/
    }

    # 5. Commit changes if there are any
    $status = git status --porcelain
    if ($status) {
        Write-Host "Committing updates..." -ForegroundColor Cyan
        git commit -m "sync: update manuscript files to match main"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to commit sync changes."
        }
    } else {
        Write-Host "No new changes to commit." -ForegroundColor Green
    }

    # 6. Push to Overleaf
    Write-Host "Pushing to Overleaf master..." -ForegroundColor Cyan
    git push overleaf overleaf-deploy:master
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to push to Overleaf. Please check your network and Overleaf repository state."
    }

    # 7. Return back to main branch
    Write-Host "Returning to main branch..." -ForegroundColor Cyan
    git checkout main
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to return to main branch."
    }

    Write-Host "Sync complete! Overleaf has been successfully updated with all your local changes." -ForegroundColor Green
}
catch {
    Write-Host ""
    Write-Host "Error during sync: $_" -ForegroundColor Red

    $currentBranch = (git branch --show-current).Trim()
    if ($currentBranch -eq "overleaf-deploy") {
        Write-Host "Aborting deploy and returning to main branch..." -ForegroundColor Yellow
        git checkout -f main
    }
    Write-Host "Sync aborted. Local files remain intact on main." -ForegroundColor Yellow
}
finally {
    # Return to the root directory
    Set-Location $rootDir
}
