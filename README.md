# Laboratory Instrument Automation Suite

Production-ready Python tools for automating laboratory test and measurement equipment. Includes responsive GUIs for Keysight oscilloscopes and Keithley instruments, reliable VISA communication, and data acquisition/export pipelines.

## Overview

The suite provides standalone GUIs and a unified launcher. It follows a modular design (GUI ↔ drivers) with background threads, queue-based status updates, and robust error handling suitable for production benches.

## Supported Instruments

- Keithley DMM6500 (DMM)
- Keithley 2230-30-1 (DC PSU)
- Keysight DSOX6004A (Oscilloscope)

## Repository Layout

```
Standalone inst controls/
├── instrument_control/          # Low-level VISA drivers and SCPI wrappers
│   ├── keithley_dmm.py
│   ├── keithley_power_supply.py
│   ├── keysight_oscilloscope.py
│   └── scpi_wrapper.py
│
├── combined_launcher.py         # Menu to launch individual GUIs
├── keithley_dmm_main.py         # DMM GUI
├── keithley_power_supply_automation.py  # PSU GUI
├── keysight_oscilloscope_main.py        # Oscilloscope GUI (with live feed)
├── unified_lab_automation_FIXED.py      # Integrated multi-instrument GUI
├── data/ graphs/ screenshots/           # Default output folders
├── requirements.txt
├── README.md
└── SETUP-INSTRUCTIONS.md
```

## Key Capabilities
- VISA-based communication (NI-VISA or Keysight IO Libraries)
- Thread-safe I/O using locks and worker threads
- Live waveform feed (oscilloscope GUI) with embedded Matplotlib
- CSV export and publication-grade plots
- Screenshot capture and full-automation workflow (screen → acquire → CSV → plot)

## System Requirements

Hardware
- Windows 10/11 (recommended), USBTMC-compatible USB ports
- NI-VISA or Keysight IO Libraries installed

Software
- Python 3.10 – 3.12 (tested). 3.14 may work if wheels are available for your platform.

## Quick Start

1) Install drivers and Python packages
```
Install NI‑VISA or Keysight IO Libraries
python -m pip install -r requirements.txt
```
2) Launch
```
python combined_launcher.py          # menu launcher
# or run a specific GUI
python keysight_oscilloscope_main.py
python keithley_dmm_main.py
python keithley_power_supply_automation.py
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
