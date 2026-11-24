<#
Makes the shortcut "Robot Car Vision", which starts the demo page with a double-click.

    tools\make_shortcut.ps1
    tools\make_shortcut.ps1 -Folder <where to put the shortcut>

Without -Folder the shortcut goes on this user's Desktop. It runs start_demo.cmd
in the project folder, which is found from where this script lies, so no path of
this laptop is written here. Running the script again replaces the shortcut.
#>
param(
    [string]$Folder = [Environment]::GetFolderPath('Desktop')
)

$ErrorActionPreference = 'Stop'

$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$target = Join-Path $root 'start_demo.cmd'
$icon = Join-Path $PSScriptRoot 'robot_car.ico'

if (-not (Test-Path -LiteralPath $target)) {
    Write-Host "start_demo.cmd was not found in $root, so no shortcut was made."
    exit 1
}
if (-not (Test-Path -LiteralPath $Folder -PathType Container)) {
    Write-Host "The folder $Folder does not exist, so no shortcut was made."
    exit 1
}

$shortcutPath = Join-Path (Resolve-Path -LiteralPath $Folder).Path 'Robot Car Vision.lnk'
if (Test-Path -LiteralPath $shortcutPath) {
    Remove-Item -LiteralPath $shortcutPath -Force    # a fresh shortcut, with nothing left from the old one
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $target
$shortcut.WorkingDirectory = $root
$shortcut.Description = 'Starts the Robot Car Vision demo page'
if (Test-Path -LiteralPath $icon) {
    $shortcut.IconLocation = "$icon,0"
}
$shortcut.Save()

Write-Host "The shortcut is at $shortcutPath"
