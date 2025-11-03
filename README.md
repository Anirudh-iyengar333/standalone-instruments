# Laboratory Instrument Automation Suite

Production-ready Python tools for automating laboratory test and measurement equipment. Includes responsive GUIs for Keysight oscilloscopes and Keithley instruments, reliable VISA communication, and data acquisition/export pipelines.

## Overview

The suite provides standalone GUIs and a unified launcher. It follows a modular design with background threads, queue-based status updates, and robust error handling suitable for production benches.

## Supported Instruments

- Keithley DMM6500 (DMM)
- Keithley 2230-30-1 (DC PSU)
- Keysight DSOX6004A (Oscilloscope)

## Repository Layout

```
Standalone inst controls/
├── scripts/
│   ├── keithley/
│   │   ├── keithley_dmm_main.py         # DMM GUI
│   │   ├── keithley_power_supply_automation.py  # PSU GUI
│   │   └── IMPROVED_SAFE_voltage_ramping_v2.3.py  # Voltage ramp utility
│   └── keysight/
│       ├── keysight_oscilloscope_main.py  # Oscilloscope GUI
│       └── keysight_oscilloscope_gradio.py  # Oscilloscope with Gradio UI
│
├── combined_launcher.py         # Unified launcher for all GUIs
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration and tooling
├── README.md                   # This file
└── SETUP-INSTRUCTIONS.md       # Detailed setup guide
```

## Key Capabilities
- VISA-based communication (NI-VISA or Keysight IO Libraries)
- Thread-safe I/O using locks and worker threads
- Live waveform feed (oscilloscope GUI) with embedded Matplotlib
- CSV export and publication-grade plots
- Screenshot capture and full-automation workflow (screen → acquire → CSV → plot)

## System Requirements

### Hardware
- Windows 10/11 (64-bit)
- USBTMC-compatible USB ports
- Minimum 4GB RAM (8GB recommended)

### Software
- Python 3.10 – 3.12 (64-bit)
- NI-VISA or Keysight IO Libraries Suite
- Required drivers for your instruments

### Python Dependencies
- Core: numpy, pandas, pyvisa
- GUI: tkinter (included with Python), matplotlib
- Additional tools in requirements-dev.txt for development

## Quick Start

1. Install the required Python packages:
```bash
# Install dependencies
python -m pip install -r requirements.txt

# Or for development with additional tools:
python -m pip install -r requirements-dev.txt
```

2. Launch the instrument control launcher:
```bash
python combined_launcher.py
```

3. Or run individual instrument GUIs directly:
```bash
# For DMM Control
python scripts/keithley/keithley_dmm_main.py

# For Power Supply Control
python scripts/keithley/keithley_power_supply_automation.py

# For Oscilloscope
python scripts/keysight/keysight_oscilloscope_main.py
```

## Configuration

VISA resource strings (examples):
```
USB0::0x0957::0x1780::MY65220169::INSTR      # Keysight DSOX6004A
USB0::0x05E6::0x6500::04561287::INSTR         # Keithley DMM6500
USB0::0x05E6::0x2230::805224014806770001::INSTR # Keithley 2230 PSU
```

Default output folders can be changed in the GUIs:
```
./data/           # CSV
./graphs/         # plots
./screenshots/    # oscilloscope images
```

## Common Tasks

Oscilloscope GUI
- Live feed: select channel → Start/Pause/Stop
- Acquire Data → Export CSV → Generate Plot
- Full Auto: screenshot → acquire → CSV → plot

Power Supply GUI
- Configure channel limits → Apply → Enable

DMM GUI
- Select function/range → Start measurement → Export/Plot

## Troubleshooting

Connection issues
- Verify VISA install and run: `python -c "import pyvisa; print(pyvisa.ResourceManager().list_resources())"`
- Try a different USB cable/port; ensure instrument not in remote lockout

Package install errors (Windows)
- Upgrade pip: `python -m pip install --upgrade pip`
- Use Python 3.10–3.12 if wheels are not available for your Python version

GUI freezes
- Long operations run in worker threads; if it freezes, check logs and console for exceptions

## Development

Code style/tooling configured via `pyproject.toml` (ruff, black). Drivers are separated under `instrument_control/`. PRs should include a short test plan and logs.

## License
All trademarks are property of their respective owners. This project is not affiliated with or endorsed by Keysight or Tektronix.

---

Last updated: 2025-10
