[app]

# (str) Title of your application
title = Crop Recommendation App

# (str) Package name
package.name = cropapp

# (str) Package domain (reverse domain notation)
package.domain = org.example

# (str) Source code where the main.py is located
source.dir = .

# (str) Main .py file
source.main = main.py

# (list) List of source files to include (like KV files, images)
source.include_exts = py,png,jpg,kv,txt

# (str) Application version
version = 1.0

# (str) Application requirements
# KivyMD, Kivy are needed; numpy/pandas not required
requirements = python3,kivy==2.2.1,kivymd==1.1.1

# (str) Icon of your app
icon.filename = icon.png

# (str) Supported orientation: portrait or landscape
orientation = portrait

# (str) Android API level to target
android.api = 33

# (str) Minimum Android SDK version required
android.minapi = 21

# (str) Android SDK version to compile against
android.sdk = 33

# (str) Android NDK version to use
android.ndk = 25b

# (str) Android NDK API level
android.ndk_api = 21

# (list) Permissions your app needs
android.permissions = INTERNET,ACCESS_NETWORK_STATE

# (str) Presplash image
presplash.filename = presplash.png

# (int) Debug mode (0=no, 1=yes)
log_level = 2

# (bool) Copy libraries instead of using .so
android.copy_libs = 1
