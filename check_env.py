import sys
import os
import importlib

print("🔍 Checking Python Environment...\n")

# 1. Check virtual environment
print("📌 Python Executable:", sys.executable)

if "venv" in sys.executable or ".venv" in sys.executable:
    print("✅ Virtual Environment is ACTIVE\n")
else:
    print("❌ Virtual Environment is NOT active\n")

# 2. Check installed packages
required_packages = [
    "flask",       # or fastapi / streamlit depending on your project
    "pandas",
    "numpy",
    "sklearn",
    "torch"
]

print("📦 Checking required packages...\n")

for pkg in required_packages:
    try:
        importlib.import_module(pkg)
        print(f"✅ {pkg} is installed")
    except ImportError:
        print(f"❌ {pkg} is NOT installed")

print()

# 3. Check project structure
required_files = [
    "app/app.py",
]

print("📁 Checking project files...\n")

for file in required_files:
    if os.path.exists(file):
        print(f"✅ Found: {file}")
    else:
        print(f"❌ Missing: {file}")

print()

# 4. Try importing your app
print("🚀 Testing app import...\n")

try:
    sys.path.append(os.getcwd())
    import app.app  # adjust if needed
    print("✅ App import successful")
except Exception as e:
    print("❌ App import failed:", e)

print()

# 5. Final status
print("🎯 Environment check complete.")