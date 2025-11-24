#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

APP_NAME="Axiom Studio"
DMG_NAME="Axiom_Studio_Installer"

echo "🚀 Starting Build Process for $APP_NAME..."

# 1. Clean previous builds
echo "🧹 Cleaning old build artifacts..."
rm -rf build dist
rm -f "$DMG_NAME.dmg"

# 2. Build the Application Bundle using PyInstaller
echo "🔨 Building .app bundle..."
pyinstaller build_studio.spec

# 3. Prepare for DMG creation
echo "📦 Preparing for packaging..."
# Create a symbolic link to /Applications inside the dist folder
# This allows the user to drag-and-drop the app easily
ln -s /Applications dist/Applications

# 4. Create the DMG
echo "💿 Creating Disk Image (.dmg)..."
hdiutil create -volname "$APP_NAME" \
    -srcfolder dist \
    -ov -format UDZO \
    "$DMG_NAME.dmg"

# 5. Cleanup the symlink
rm dist/Applications

echo "✅ Success! Installer created:"
echo "   $PWD/$DMG_NAME.dmg"