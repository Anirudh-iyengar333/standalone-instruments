# Setup Instructions for Standalone Instrument Controls

This guide explains how to install and run the Laboratory Instrument Automation Suite on Windows. It assumes basic familiarity with PowerShell.

## Prerequisites

Hardware
- Windows 10/11 PC with USB ports
- Instruments connected via USB (USBTMC)

Software
- NI‑VISA Runtime or Keysight IO Libraries Suite
- Python 3.10–3.12 (64‑bit)

---

## 1) Install Python (3.10–3.12)

Download from https://www.python.org/downloads/windows/ and during setup check “Add Python to PATH”.
Verify:
```powershell
python --version
```

---

## 2) Install VISA Drivers

Choose one:
- NI‑VISA Runtime: https://www.ni.com/en/support/downloads/drivers/download.ni-visa.html
- Keysight IO Libraries: https://www.keysight.com/find/iosuite
Reboot if requested.

---

## 3) Get the project

Open PowerShell and change to your workspace folder (example: Desktop), then clone or copy the project files.

---

## 4) Create a virtual environment (recommended)

From the project root:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

To deactivate later: `deactivate`

---

## 5) Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify VISA is accessible:
```powershell
python -c "import pyvisa as v; print(v.ResourceManager().list_resources())"
```

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
pip install -r requirements.txt --upgrade
```

Last updated: 2025-10
