# Setup Instructions for Standalone Instrument Controls

This guide explains how to install and run the Laboratory Instrument Automation Suite on Windows. It assumes basic familiarity with command-line interfaces.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Install Python](#1-install-python-310312)
3. [Install VISA Drivers](#2-install-visa-drivers)
4. [Set Up the Project](#3-set-up-the-project)
5. [Install Dependencies](#4-install-dependencies)
6. [Running the Application](#5-running-the-application)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware
- Windows 10/11 PC (64-bit) with USB ports
- Instruments connected via USB (USBTMC)
- Minimum 4GB RAM (8GB recommended)

### Software
- NI‑VISA Runtime or Keysight IO Libraries Suite
- Python 3.10–3.12 (64‑bit)
- Git (recommended for development)

---

## 1. Install Python (3.10–3.12)

1. Download the latest Python 3.12.x installer from [python.org](https://www.python.org/downloads/windows/)
2. **Important**: During installation, check "Add Python to PATH"
3. Verify the installation:
   ```powershell
   python --version
   pip --version
   ```

---

## 2. Install VISA Drivers

Choose one of the following VISA implementations:

### Option A: NI-VISA (Recommended)
1. Download from [NI-VISA](https://www.ni.com/en/support/downloads/drivers/download.ni-visa.html)
2. Run the installer with default settings
3. Reboot your computer if prompted

### Option B: Keysight IO Libraries
1. Download from [Keysight](https://www.keysight.com/find/iosuite)
2. Run the installer with default settings
3. Reboot your computer if prompted

---

## 3. Set Up the Project

### Option A: Using Git (Recommended for Development)
```powershell
# Clone the repository
git clone <repository-url>
cd <project-folder>
```

### Option B: Manual Download
1. Download the project as a ZIP file
2. Extract to your preferred location
3. Open a terminal in the extracted folder

---

## 4. Create a Virtual Environment

### Windows
```powershell
# Create a virtual environment
python -m venv .venv

# Activate the environment
.venv\Scripts\Activate
```

### Linux/macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 5. Install Dependencies

With the virtual environment activated:

```powershell
# First upgrade pip
python -m pip install --upgrade pip

# Install core requirements
python -m pip install -r requirements.txt
```

Verify VISA is accessible:
```powershell
python -c "import pyvisa as v; print(v.ResourceManager().list_resources())"
```

> **Note**: Using `python -m pip` is more reliable than just `pip` as it ensures you're using the pip from the correct Python installation, especially when working with multiple Python versions or virtual environments.

---

## 6) Run the applications

From the project root:
```powershell
python combined_launcher.py                 # menu launcher
# or run GUIs directly
python keysight_oscilloscope_main.py
python keithley_dmm_main.py
python keithley_power_supply_automation.py
```

---

## Folder locations

Default output paths (change in GUI if needed):
```
./data/          # CSV
./graphs/        # plots
./screenshots/   # instrument screenshots
```

---

## Troubleshooting

No instruments found
- Reinstall/repair VISA
- Try another USB cable/port; power-cycle instrument
- Check Windows Device Manager for USBTMC devices

Package installation errors
- Upgrade pip: `python -m pip install --upgrade pip`
- Use Python 3.10–3.12 if a dependency wheel is missing for your version

GUI appears but buttons are disabled
- Connect to the instrument first; operations enable on successful connection

---

## Updating

If the repository updates:
```powershell
python -m pip install -r requirements.txt --upgrade
```

Last updated: 2025-10
