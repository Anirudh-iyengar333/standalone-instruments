# Standalone Instrument Controls

> Professional laboratory automation suite for Keysight oscilloscopes, Keithley DMMs, and power supplies

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![VISA](https://img.shields.io/badge/VISA-NI%20%7C%20Keysight-orange)](https://www.ni.com/en/support/downloads/drivers/download.ni-visa.html)

## Overview

Standalone Instrument Controls is a comprehensive Python-based automation framework for laboratory instrumentation. Designed for reliability and ease of use, it provides both graphical and web-based interfaces for controlling test and measurement equipment via USB/VISA communication protocols.

### Key Features

- **Multi-Instrument Support**: Keysight DSOX6004A oscilloscopes, Keithley DMM6500/DMM7510 multimeters, Keithley 2280S/2231A power supplies
- **Multiple Interface Options**: Traditional GUI (tkinter), live waveform display, and modern web interface (Gradio)
- **Comprehensive Data Acquisition**: High-resolution waveform capture (up to 62,500 points), automated measurements, and statistical analysis
- **Professional Export Capabilities**: CSV with metadata, publication-quality plots, PNG screenshots
- **Workflow Automation**: Single-click workflows combining screenshot capture, data acquisition, CSV export, and plot generation
- **Robust Error Handling**: Comprehensive logging, connection recovery, and detailed status reporting
- **Cross-Platform VISA Support**: Compatible with NI-VISA and Keysight IO Libraries

### Supported Instruments

| Instrument | Model | Communication |
|------------|-------|---------------|
| Oscilloscope | Keysight DSOX6004A | USB (USBTMC) |
| Digital Multimeter | Keithley DMM6500, DMM7510 | USB (USBTMC) |
| Power Supply | Keithley 2280S-60-3, 2231A-30-3 | USB (USBTMC) |

## Quick Start

### Prerequisites

- **Operating System**: Windows 10/11 (64-bit), Linux, or macOS
- **Python**: 3.10, 3.11, or 3.12 (64-bit recommended)
- **VISA Drivers**: NI-VISA Runtime or Keysight IO Libraries Suite
- **Hardware**: USB ports, 4GB RAM minimum (8GB recommended for live waveform display)

### Installation

```bash
**Open Command Prompt and run:**
cd “your preferred location”

# Clone the repository
git clone https://github.com/anirudhiyengar-cell/Standalone_instrument_controls.git
cd "your file loaction/Standalone_instrument_controls"

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Verify VISA installation
python -c "import pyvisa; print(pyvisa.ResourceManager().list_resources())"
```

### Running the Applications

```bash
# Launch unified menu (recommended)
python scripts/combined_launcher.py

# Or run specific instruments directly
python scripts/keysight/keysight_oscilloscope_main.py
python scripts/keithley/keithley_dmm_main.py
python scripts/keithley/keithley_power_supply_automation.py
```

## Project Structure

```
Standalone_instrument_controls/
├── scripts/
│   ├── keysight/              # Keysight oscilloscope applications
│   │   ├── keysight_oscilloscope_main.py
│   │   ├── keysight_oscilloscope_main_with_livefeed.py
│   │   └── keysight_oscilloscope_gradio.py
│   ├── keithley/              # Keithley instrument applications
│   │   ├── keithley_dmm_main.py
│   │   ├── keithley_power_supply_automation.py
│   │   └── IMPROVED_SAFE_voltage_ramping_v2.3.py
│   └── combined_launcher.py   # Unified application launcher
├── instrument_control/        # Core instrument control modules
├── data/                      # CSV data exports (auto-created)
├── graphs/                    # Plot outputs (auto-created)
├── screenshots/               # Instrument screenshots (auto-created)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── SETUP-INSTRUCTIONS.md      # Detailed setup guide
└── LICENSE                    # License information
```

## Documentation

- **[Setup Instructions](SETUP-INSTRUCTIONS.md)**: Comprehensive installation and configuration guide
- **[Troubleshooting](SETUP-INSTRUCTIONS.md#troubleshooting)**: Common issues and solutions
- **[User Guide](docs/USER_GUIDE.md)**: Detailed usage instructions *(coming soon)*
- **[API Reference](docs/API.md)**: Module and class documentation *(coming soon)*

## Usage Examples

### Oscilloscope: Capture and Analyze Waveform

```python
from instrument_control.keysight_oscilloscope import KeysightOscilloscope

# Connect to oscilloscope
osc = KeysightOscilloscope("USB0::0x0957::0x1780::MY65220169::INSTR")
osc.connect()

# Configure channel
osc.configure_channel(channel=1, v_div=1.0, offset=0.0, coupling='DC')

# Set timebase
osc.set_timebase(time_div='1ms', offset=0.0)

# Acquire waveform
time_data, voltage_data = osc.acquire_waveform(channel=1)

# Export to CSV
osc.export_to_csv(time_data, voltage_data, filename='output.csv')
```

### DMM: Automated Voltage Measurement

```python
from instrument_control.keithley_dmm import KeithleyDMM

# Connect to multimeter
dmm = KeithleyDMM("USB0::0x05E6::0x6500::04547164::INSTR")
dmm.connect()

# Configure for DC voltage measurement
dmm.configure_dc_voltage(range=10.0, nplc=1.0)

# Read measurement
voltage = dmm.measure()
print(f"Measured voltage: {voltage:.6f} V")
```

### Power Supply: Safe Voltage Ramping

```python
from instrument_control.keithley_power_supply import KeithleyPowerSupply

# Connect to power supply
psu = KeithleyPowerSupply("USB0::0x05E6::0x2280::9205274::INSTR")
psu.connect()

# Ramp voltage safely
psu.ramp_voltage(channel=1, target_voltage=12.0, step_size=0.1, step_delay=0.05)

# Set current limit
psu.set_current_limit(channel=1, current=2.0)
```

## Features by Application

### Keysight Oscilloscope

- **Channel Configuration**: V/div, offset, coupling (AC/DC), probe attenuation (1x/10x/100x)
- **Timebase Control**: Time/div from 1ns to 50s, horizontal offset
- **Trigger Configuration**: Source selection, level, slope (rising/falling), autoscale
- **Function Generators**: WGEN1/WGEN2 with multiple waveforms (sine, square, ramp, pulse, DC, noise)
- **Data Acquisition**: Up to 62,500 points per channel, simultaneous multi-channel capture
- **Measurements**: Automated frequency, period, amplitude, RMS, peak-to-peak
- **Export Options**: CSV with metadata, PNG plots with measurements, oscilloscope screenshots
- **Live Waveform Display**: Real-time waveform visualization (updated GUI version)

### Keithley DMM

- **Measurement Functions**: DCV, ACV, DCI, ACI, 2-wire/4-wire resistance, continuity, diode test
- **Configuration**: Range selection, integration time (NPLC), auto-zero, filtering
- **High-Speed Digitizer**: Waveform capture mode with configurable sample rate
- **Data Logging**: Timestamped measurements, statistics (min/max/avg/stdev)
- **Export**: CSV format with full configuration metadata

### Keithley Power Supply

- **Multi-Channel Control**: Independent control of all output channels
- **Safe Ramping**: Configurable voltage ramping with step size and delay
- **Current/Voltage Limits**: Programmable limits with real-time monitoring
- **Output Protection**: Over-voltage and over-current protection
- **Measurement Readback**: Real-time voltage and current monitoring

## Configuration

Default output directories can be modified in the GUI or by editing configuration files:

```python
# Default paths (relative to project root)
DATA_DIR = "./data"          # CSV exports
GRAPHS_DIR = "./graphs"      # Plot images
SCREENSHOTS_DIR = "./screenshots"  # Instrument screenshots
```

## Troubleshooting

### Common Issues

**1. PowerShell Script Execution Disabled**
```powershell
# Solution: Use Command Prompt instead
.venv\Scripts\activate.bat

# Or enable PowerShell scripts (run as Administrator)
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**2. File Not Found Errors**
```bash
# Ensure you're in the project root directory
cd Standalone_instrument_controls

# Run scripts with correct relative paths
python scripts/keysight/keysight_oscilloscope_main.py
```

**3. VISA/Instrument Not Found**
```python
# Check available instruments
python -c "import pyvisa; rm = pyvisa.ResourceManager(); print(rm.list_resources())"

# Common solutions:
# - Reinstall VISA drivers (NI-VISA or Keysight IO Libraries)
# - Power cycle the instrument
# - Try different USB port/cable
# - Check Windows Device Manager for USBTMC devices
```

**4. Module Import Errors**
```bash
# Ensure virtual environment is activated (look for (.venv) in prompt)
# Reinstall dependencies
python -m pip install -r requirements.txt --upgrade
```

See [SETUP-INSTRUCTIONS.md](SETUP-INSTRUCTIONS.md#troubleshooting) for comprehensive troubleshooting.

## Development

### Running Tests

```bash
# Install development dependencies
python -m pip install -r requirements-dev.txt

# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=instrument_control tests/
```

### Code Style

This project follows PEP 8 guidelines with Black formatting:

```bash
# Format code
black .

# Check style
flake8 instrument_control/ scripts/

# Type checking
mypy instrument_control/
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass
- Documentation is updated
- Commit messages are descriptive

## Roadmap

- [ ] Add support for additional Keysight oscilloscope models
- [ ] Implement automated test sequences
- [ ] Add waveform analysis tools (FFT, filtering)
- [ ] Create standalone executables (PyInstaller)
- [ ] Expand documentation with video tutorials
- [ ] Add data visualization dashboard
- [ ] Implement remote instrument control over network

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [PyVISA](https://pyvisa.readthedocs.io/) for instrument communication
- GUI framework: [tkinter](https://docs.python.org/3/library/tkinter.html) and [Gradio](https://gradio.app/)
- Data processing: [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- Visualization: [Matplotlib](https://matplotlib.org/)

## Support

For issues, questions, or contributions:

- **Issues**: [GitHub Issues](https://github.com/anirudhiyengar-cell/Standalone_instrument_controls/issues)
- **Discussions**: [GitHub Discussions](https://github.com/anirudhiyengar-cell/Standalone_instrument_controls/discussions)
- **Email**: [your-email@example.com](mailto:your-email@example.com)

## Citation

If you use this software in your research, please cite:

```bibtex
@software{standalone_instrument_controls,
  author = {Anirudh Iyengar},
  title = {Standalone Instrument Controls: Professional Laboratory Automation Suite},
  year = {2025},
  url = {https://github.com/anirudhiyengar-cell/Standalone_instrument_controls}
}
```

---
 
**Version**: 1.0  
**Maintainer**: Anirudh Iyengar  
**Status**: Production Ready