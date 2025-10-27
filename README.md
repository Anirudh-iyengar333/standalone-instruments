# Laboratory Instrument Automation Suite

A professional-grade Python automation framework for laboratory test and measurement equipment, designed for scientific research and industrial testing environments. This suite provides comprehensive control interfaces for Keithley and Keysight instruments with advanced data acquisition, analysis, and visualization capabilities.

## Overview

This project implements a complete automation solution for coordinated control of multiple laboratory instruments through a unified interface. Built with industry-standard practices, the system features multi-threaded architecture, VISA-compliant communication, and robust error handling suitable for production test environments.

## Supported Instruments

- **Keithley DMM6500** - 6½-digit bench/system digital multimeter
- **Keithley 2230-30-1** - Triple-channel programmable DC power supply (30V, 3A per channel)
- **Keysight DSOX6004A** - 1 GHz mixed-signal oscilloscope (4 analog + 16 digital channels)

## Architecture

The system employs a modular architecture with clear separation of concerns:

```
project_root/
├── instrument_control/          # Low-level VISA instrument drivers
│   ├── keithley_dmm.py         # DMM6500 SCPI wrapper
│   ├── keithley_power_supply.py # PSU command interface
│   ├── keysight_oscilloscope.py # Oscilloscope control API
│   └── scpi_wrapper.py          # Base SCPI communication layer
│
├── combined_launcher.py         # Main application entry point
├── keithley_dmm_main.py        # DMM automation GUI
├── keithley_power_supply_automation.py  # PSU control GUI
├── keysight_oscilloscope_main.py        # Oscilloscope GUI
├── IMPROVED_SAFE_voltage_ramping_v2.3.py  # Automated ramping system
└── unified_lab_automation_FIXED.py      # Integrated multi-instrument control
```

**Design Principles:**
- Thread-safe queue-based inter-process communication
- Non-blocking GUI operations with background worker threads
- Stateless instrument command wrappers for reliability
- Comprehensive logging and error recovery mechanisms

## Features

### Digital Multimeter (DMM6500)
- **8 Measurement Functions**: DC/AC voltage, DC/AC current, 2-wire/4-wire resistance, capacitance, frequency
- **High Precision**: 9-digit resolution with configurable NPLC (integration time)
- **Acquisition Modes**: Single shot, continuous polling, statistical analysis
- **Data Export**: Timestamped CSV files with measurement metadata
- **Visualization**: Real-time plotting with embedded statistics (mean, std dev, min/max)

### Power Supply (2230-30-1)
- **Multi-Channel Control**: Independent configuration of 3 output channels
- **Output Ranges**: 0-30V, 0-3A per channel with programmable OVP/OCP limits
- **Waveform Generation**: Arbitrary voltage profiles (sine, square, triangle, ramp)
- **Real-Time Monitoring**: Simultaneous voltage, current, and power readback
- **Safety Features**: Emergency stop, current limiting, sequential channel enable/disable
- **Automated Testing**: Programmable ramping cycles with configurable timing

### Oscilloscope (DSOX6004A)
- **Waveform Acquisition**: Multi-channel data capture with configurable sample depth (up to 62,500 points)
- **Timebase Control**: Programmable horizontal scale (1 ns/div to 50 s/div)
- **Trigger Configuration**: Edge triggering with adjustable level, slope, and source selection
- **Function Generators**: WGEN1/WGEN2 output configuration (sine, square, ramp, pulse, DC, noise)
- **Screenshot Capture**: Direct instrument display image export
- **Data Analysis**: Automatic voltage statistics (peak, RMS, mean, standard deviation)

### Voltage Ramping System
- **Synchronized Operation**: Coordinated PSU voltage control with DMM measurement feedback
- **Waveform Library**: Sine, square, triangle, ramp-up, ramp-down with configurable amplitude/frequency
- **Multi-Cycle Testing**: Automated repetition with per-cycle statistics
- **Comparative Analysis**: Set-point vs. measured voltage tracking with error quantification
- **Visual Feedback**: Real-time progress indicators and ETA calculation
- **Comprehensive Reporting**: Combined graphs with error analysis and statistical summaries

## System Requirements

### Software Dependencies
```
Python 3.7 or higher
PyVISA >= 1.11.3
numpy >= 1.19.0
pandas >= 1.1.0
matplotlib >= 3.3.0
```

### Hardware Requirements
- **VISA Backend**: NI-VISA, Keysight IO Libraries, or compatible VISA implementation
- **Connectivity**: USB 2.0 or higher (USBTMC protocol support)
- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 10.14+

### Installation

1. **Install Python Dependencies**
```bash
pip install pyvisa numpy pandas matplotlib
```

2. **Install VISA Drivers**
   - **Windows/macOS**: [NI-VISA Runtime](https://www.ni.com/en-us/support/downloads/drivers/download.ni-visa.html)
   - **Linux**: [Keysight IO Libraries](https://www.keysight.com/us/en/lib/software-detail/computer-software/io-libraries-suite-downloads-2175637.html)

3. **Verify Installation**
```bash
python -m visa info
```

4. **Connect Instruments** via USB and verify device enumeration:
```bash
python -m visa list
```

## Quick Start

### Launching the Application

**Unified Launcher** (Recommended):
```bash
python combined_launcher.py
```
This presents a menu to launch individual instrument GUIs or the integrated multi-instrument interface.

**Direct Application Launch**:
```bash
python keithley_dmm_main.py              # DMM standalone
python keithley_power_supply_automation.py  # PSU standalone
python keysight_oscilloscope_main.py        # Oscilloscope standalone
python unified_lab_automation_FIXED.py      # Multi-instrument interface
```

### Basic Usage Example

**DMM Measurement Workflow**:
1. Launch DMM application
2. Enter VISA address (e.g., `USB0::0x05E6::0x6500::04561287::INSTR`)
3. Click **Connect** and verify instrument identification
4. Select measurement function (e.g., DC Voltage)
5. Configure range and resolution
6. Click **Single Measurement** or **Start Continuous**
7. Export data via **Export to CSV** or generate plots with **Generate Graph**

**Power Supply Configuration**:
1. Launch PSU application
2. Connect to instrument using VISA address
3. Configure each channel:
   - Set voltage (0-30V)
   - Set current limit (0-3A)
   - Configure OVP threshold
4. Click **Configure** then **Enable** for each channel
5. Monitor real-time V/I/P measurements
6. Use **Export Data** to log measurements to CSV

**Voltage Ramping Test**:
1. Launch voltage ramping application
2. Connect PSU and DMM
3. Configure waveform parameters:
   - Waveform type (Sine/Square/Triangle/Ramp)
   - Target voltage (max 5V for safety)
   - Number of cycles
   - Points per cycle
   - Cycle duration
4. Click **Start Safe Ramping**
5. Monitor progress bar and real-time voltage display
6. Review combined analysis graphs upon completion

## Configuration

### VISA Address Format
Instruments communicate via VISA resource strings. Common formats:
- USB: `USB0::<vendor_id>::<product_id>::<serial>::INSTR`
- TCPIP: `TCPIP0::<ip_address>::<port>::SOCKET`
- GPIB: `GPIB0::<primary_address>::INSTR`

**Example VISA Addresses** (update serial numbers for your instruments):
```python
DMM6500:     "USB0::0x05E6::0x6500::04561287::INSTR"
PSU 2230:    "USB0::0x05E6::0x2230::805224014806770001::INSTR"
DSOX6004A:   "USB0::0x0957::0x1780::MY65220169::INSTR"
```

### Default File Locations
Data files are saved to subdirectories in the current working directory:
- **CSV Data**: `./dmm_data/`, `./voltage_ramp_data/`
- **Graphs**: `./dmm_graphs/`, `./voltage_ramp_graphs/`
- **Screenshots**: `./oscilloscope_screenshots/`

These paths can be customized through the GUI file preferences panels.

## Data Export Formats

### CSV Structure
All exported CSV files include metadata headers:
```
# Keithley DMM6500 Measurement Data
# Export Time: 2025-10-27T10:30:45.123456
# Total Measurements: 100
# Columns: timestamp, measurement_type, value, unit

timestamp,measurement_type,value,unit
2025-10-27T10:28:15.234567,DC Voltage,1.234567,V
2025-10-27T10:28:16.345678,DC Voltage,1.234890,V
...
```

### Graph Output
Graphs are exported as PNG images (300 DPI) with embedded statistics boxes showing:
- Data point count
- Mean, standard deviation
- Minimum, maximum values
- Range (max - min)

## Safety Features

### Power Supply Protections
- **Emergency Stop**: Immediate output disable for all channels
- **Over-Voltage Protection (OVP)**: Configurable per-channel voltage limits
- **Over-Current Protection (OCP)**: Automatic current limiting
- **Soft Start**: Gradual voltage ramping to prevent inrush damage

### Voltage Ramping Safety
- **Hard Voltage Limit**: Software-enforced 5V maximum for automated ramping
- **Ramp Rate Limiting**: Configurable settling time between voltage steps
- **Real-Time Monitoring**: Continuous comparison of set vs. measured voltage
- **Abort Capability**: Instant ramping termination with safe shutdown sequence

## Troubleshooting

### Connection Issues
**Problem**: "VISA resource not found" error

**Solutions**:
- Verify USB cable connection and try different USB port
- Run `python -m visa list` to enumerate available devices
- Check instrument is powered on and not in remote lockout mode
- Reinstall VISA drivers (NI-VISA or Keysight IO Libraries)
- For Linux, ensure user has permissions: `sudo usermod -a -G dialout $USER`

### Measurement Errors
**Problem**: DMM returns `9.9E+37` (overflow)

**Solutions**:
- Check input range selection (use auto-range or higher range)
- Verify measurement function matches signal type
- Inspect test lead connections
- Check for open/short circuit conditions

### GUI Freezing
**Problem**: Application becomes unresponsive during operations

**Solutions**:
- Operations run in background threads and should not block GUI
- If issue persists, check Python threading compatibility
- Monitor system resources (CPU/memory)
- Review application log files for exceptions

### Data Export Failures
**Problem**: CSV export fails or produces empty files

**Solutions**:
- Verify write permissions on target directory
- Ensure sufficient disk space
- Check for invalid characters in custom filenames
- Confirm data collection has occurred before export

## Development

### Code Style
This project follows PEP 8 conventions with comprehensive inline documentation:
- Google-style docstrings for all classes and methods
- Type hints for function signatures (Python 3.7+ typing module)
- Detailed inline comments explaining instrument-specific behavior

### Architecture Patterns
- **Observer Pattern**: GUI subscribes to instrument state changes via queues
- **Command Pattern**: Instrument operations encapsulated as discrete methods
- **Factory Pattern**: Instrument drivers instantiated through wrapper classes

### Extending Functionality

**Adding New Measurement Functions**:
1. Define new function in `MEASUREMENT_OPTIONS` constant
2. Implement corresponding SCPI commands in driver wrapper
3. Update GUI dropdown menu values
4. Add unit conversion logic if required

**Integrating Additional Instruments**:
1. Create new driver in `instrument_control/` package
2. Inherit from `SCPIWrapper` base class
3. Implement instrument-specific command methods
4. Create standalone GUI application or integrate into unified interface

## Contributing

While this is a professional engineering project, contributions are welcome:
- Report issues through detailed bug reports with log files
- Propose enhancements with clear use case descriptions
- Submit pull requests following existing code style conventions
- Provide instrument-specific test cases for validation

## License

This project is provided as-is for laboratory automation purposes. All trademarks (Keithley, Keysight) are property of their respective owners. This software is not affiliated with or endorsed by Tektronix or Keysight Technologies.

## Support

For technical support:
- Review application log files (generated in working directory)
- Check instrument programming manuals for SCPI command references
- Verify VISA driver installation and compatibility
- Consult instrument front panel for error codes

**Instrument Manuals**:
- [DMM6500 Programming Manual](https://www.tek.com/keithley-digital-multimeter/dmm6500-manual)
- [2230-30-1 User Manual](https://www.tek.com/en/manual/dc-power-supply/series-2230-triple-channel-power-supplies-users-manual-2220-2230-2231-series)
- [DSOX6004A Programmer's Guide](https://www.keysight.com/us/en/product/DSOX6004A/oscilloscope-1-ghz-4-analog-channels.html)

## Version History

**Current Version**: 2.0 (October 2025)

**Features**:
- Multi-threaded architecture for responsive GUI operation
- Comprehensive error handling and logging
- Publication-quality data export and visualization
- Safety-critical voltage ramping with real-time monitoring
- Unified launcher for streamlined workflow

**Tested Instruments**:
- Keithley DMM6500 (Firmware: 1.7.10c)
- Keithley 2230-30-1 (Firmware: 1.19)
- Keysight DSOX6004A (Firmware: 07.50.2021102830)

---

**Author**: Professional Instrumentation Control System  
**Last Updated**: October 2025  
**Status**: Production Ready
