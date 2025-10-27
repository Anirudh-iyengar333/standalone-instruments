# Setup Instructions for Standalone Instrument Controls

This guide will help you set up and run the laboratory instrument automation scripts on any Windows computer.

## Prerequisites

Before starting, ensure you have:
- A Windows computer with Administrator access
- Internet connection for downloading software
- Laboratory instruments (Keithley Power Supply, Oscilloscope, DMM) connected via USB

---

## Step 1: Install Git

1. Download Git from: https://git-scm.com/download/win
2. Run the installer with default settings
3. Open PowerShell and verify installation:
   ```powershell
   git --version
   ```
   You should see something like: `git version 2.42.0.windows.1`

---

## Step 2: Install Python

1. Download Python 3.11+ from: https://www.python.org/downloads/
2. **IMPORTANT**: During installation, check ☑ "Add Python to PATH"
3. Run the installer with default settings
4. Open PowerShell and verify installation:
   ```powershell
   python --version
   ```
   You should see: `Python 3.11.x` or higher

---

## Step 3: Install NI-VISA Drivers (Required for Instrument Communication)

1. Download NI-VISA from: https://www.ni.com/en-us/support/downloads/drivers/download.ni-visa.html
2. Install with default settings
3. Restart your computer after installation

---

## Step 4: Clone the Repository from GitHub

1. Open PowerShell
2. Navigate to where you want to download the project (e.g., Desktop):
   ```powershell
   cd C:\Users\YourUsername\Desktop
   ```

3. Clone the repository:
   ```powershell
   git clone https://github.com/anirudhiyengar-cell/Standalone_instrument_controls.git
   ```

4. Navigate into the project folder:
   ```powershell
   cd Standalone_instrument_controls
   ```

---

## Step 5: Install Python Dependencies

1. Install all required Python packages:
   ```powershell
   pip install -r requirements.txt
   ```

2. Wait for installation to complete (2-3 minutes)

3. Verify installation:
   ```powershell
   pip list
   ```
   You should see: pyvisa, numpy, matplotlib, Pillow, etc.

---

## Step 6: Connect Your Instruments

1. Connect instruments to computer via USB
2. Power on all instruments
3. Wait for Windows to recognize devices (drivers may install automatically)

---

## Step 7: Verify Instrument Connection

1. Run the test script to find connected instruments:
   ```powershell
   python -c "import pyvisa; rm = pyvisa.ResourceManager(); print(rm.list_resources())"
   ```

2. You should see output like:
   ```
   ('USB0::0x05E6::0x2450::04461234::INSTR', 'USB0::0x0957::0x1796::MY12345678::INSTR')
   ```

3. If no instruments appear:
   - Check USB connections
   - Ensure NI-VISA is installed
   - Restart computer and try again

---

## Step 8: Run the Instrument Control Scripts

### For Keithley Power Supply Control:
```powershell
python keithley_power_supply.py
```

### For Oscilloscope Control:
```powershell
python keysight_oscilloscope.py
```

### For Digital Multimeter Control:
```powershell
python keithley_dmm.py
```

---

## Step 9: Using the GUI

1. The GUI window will appear
2. Configure instrument settings in the interface
3. Click "Connect" to establish communication
4. Use control buttons to operate the instrument
5. Click "Disconnect" when finished

---

## Troubleshooting

### Problem: "python: command not found"
**Solution**: Python not in PATH. Reinstall Python and check "Add to PATH" option.

### Problem: "No module named 'pyvisa'"
**Solution**: Run: `pip install -r requirements.txt`

### Problem: "No instruments found"
**Solution**: 
1. Check USB connections
2. Install NI-VISA drivers
3. Restart computer
4. Verify instruments are powered on

### Problem: "Access denied" or "Permission error"
**Solution**: Run PowerShell as Administrator:
- Right-click PowerShell → "Run as Administrator"

### Problem: Git clone fails
**Solution**: Check internet connection and GitHub URL is correct

---

## Updating to Latest Version

If the code has been updated on GitHub:

```powershell
# Navigate to project folder
cd C:\Users\YourUsername\Desktop\Standalone_instrument_controls

# Pull latest changes
git pull origin main

# Reinstall dependencies (if requirements changed)
pip install -r requirements.txt
```

---

## Complete Command Summary (Quick Reference)

```powershell
# 1. Navigate to Desktop
cd C:\Users\YourUsername\Desktop

# 2. Clone repository
git clone https://github.com/anirudhiyengar-cell/Standalone_instrument_controls.git

# 3. Enter project folder
cd Standalone_instrument_controls

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the script
python keithley_power_supply.py
```

---

## System Requirements

- **Operating System**: Windows 10/11
- **Python Version**: 3.11 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Disk Space**: 500MB for software and dependencies
- **USB Ports**: Available for instrument connections

---

## Support

For issues or questions:
- Check the troubleshooting section above
- Review instrument manuals for SCPI commands
- Verify all prerequisites are installed correctly

---

**Last Updated**: October 27, 2025
**Project Repository**: https://github.com/anirudhiyengar-cell/Standalone_instrument_controls
