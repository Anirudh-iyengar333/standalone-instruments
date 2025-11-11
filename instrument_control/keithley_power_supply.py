#!/usr/bin/env python3
"""
Keithley 2230 Series Power Supply Control Library
==================================================

Professional SCPI Command Wrapper for Keithley 2230/2231 Triple-Channel Power Supplies

This module provides a comprehensive Python interface to control Keithley 2230 series
triple-channel power supplies via VISA communication. It implements the full SCPI 
command set as documented in the 2230-900-01B user manual (April 2022).

Features:
    - Complete SCPI command coverage for all power supply functions
    - Robust error handling and connection management
    - Type-safe interfaces with dataclasses and enums
    - Comprehensive logging for debugging and audit trails
    - Support for channel combination modes (tracking, series, parallel)
    - Timer-based output control
    - Save/recall functionality for setup configurations
    - Status register monitoring and event handling
    - Voltage/current step adjustment capabilities
    - Power measurement and monitoring

Supported Models:
    - Keithley 2230-30-3: 3 channels, 30V/3A per channel
    - Keithley 2231A-30-3: 3 channels, 30V/3A per channel
    - Keithley 2280S: 1 channel, 72V/120A (single channel model)

Author: Instrumentation Control Team
Version: 2.0.0
Date: 2024
License: MIT

Dependencies:
    - pyvisa: VISA library for instrument communication
    - Python 3.7+

Usage Example:
    >>> from keithley_power_supply import KeithleyPowerSupply
    >>> psu = KeithleyPowerSupply("USB0::0x05E6::0x2230::INSTR")
    >>> psu.connect()
    >>> psu.configure_channel(1, voltage=5.0, current_limit=1.0, ovp_level=6.0)
    >>> psu.enable_channel_output(1)
    >>> voltage, current = psu.measure_channel_output(1)
    >>> psu.disconnect()
"""

import logging
import time
import re
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum
import pyvisa
from pyvisa.errors import VisaIOError

# ============================================================================
# EXCEPTION CLASSES
# ============================================================================

class KeithleyPowerSupplyError(Exception):
    """
    Base exception class for Keithley power supply errors.

    Raised when operations fail due to communication errors, invalid parameters,
    or instrument-specific issues.
    """
    pass

# ============================================================================
# ENUMERATION CLASSES
# ============================================================================

class OutputState(Enum):
    """
    Enumeration for output enable/disable states.

    Attributes:
        ENABLED: Output is active and supplying power
        DISABLED: Output is off
        UNKNOWN: State could not be determined
    """
    ENABLED = "ENABLED"
    DISABLED = "DISABLED"
    UNKNOWN = "UNKNOWN"

class ProtectionState(Enum):
    """
    Enumeration for protection circuit states.

    Attributes:
        NORMAL: No protection triggered, normal operation
        OVP_TRIPPED: Over-voltage protection activated
        OCP_TRIPPED: Over-current protection activated
        UNKNOWN: Protection state could not be determined
    """
    NORMAL = "NORMAL"
    OVP_TRIPPED = "OVP_TRIPPED"
    OCP_TRIPPED = "OCP_TRIPPED"
    UNKNOWN = "UNKNOWN"

class CombinationMode(Enum):
    """
    Enumeration for channel combination modes.

    Attributes:
        NONE: Channels operate independently
        TRACK_CH1_CH2: CH1 and CH2 track (same voltage/current settings)
        TRACK_CH2_CH3: CH2 and CH3 track
        TRACK_ALL: All three channels track
        SERIES_CH1_CH2: CH1 and CH2 in series (voltages add)
        PARALLEL_CH1_CH2: CH1 and CH2 in parallel (currents add)
        PARALLEL_CH2_CH3: CH2 and CH3 in parallel
        PARALLEL_ALL: All three channels in parallel
    """
    NONE = "NONE"
    TRACK_CH1_CH2 = "TRACK_CH1CH2"
    TRACK_CH2_CH3 = "TRACK_CH2CH3"
    TRACK_ALL = "TRACK_CH1CH2CH3"
    SERIES_CH1_CH2 = "SERIES_CH1CH2"
    PARALLEL_CH1_CH2 = "PARALLEL_CH1CH2"
    PARALLEL_CH2_CH3 = "PARALLEL_CH2CH3"
    PARALLEL_ALL = "PARALLEL_CH1CH2CH3"

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ChannelConfiguration:
    """
    Data class representing a channel's complete configuration.

    Attributes:
        channel: Channel number (1, 2, or 3)
        voltage: Set voltage value in volts
        current_limit: Current limit in amperes
        ovp_level: Over-voltage protection threshold in volts
        output_enabled: Whether the output is currently enabled
    """
    channel: int
    voltage: float
    current_limit: float
    ovp_level: float
    output_enabled: bool

@dataclass
class ChannelMeasurement:
    """
    Data class representing measured values from a channel.

    Attributes:
        channel: Channel number (1, 2, or 3)
        voltage: Measured voltage in volts
        current: Measured current in amperes
        power: Calculated power in watts
        output_state: Current output enable state
        protection_state: Current protection circuit state
    """
    channel: int
    voltage: float
    current: float
    power: float
    output_state: OutputState
    protection_state: ProtectionState

@dataclass
class SystemStatus:
    """
    Data class representing the instrument's system status.

    Attributes:
        error_code: Most recent error code (0 = no error)
        error_message: Human-readable error description
        operation_complete: Whether all pending operations are complete
        questionable_status: Questionable status register value
        standard_event_status: Standard event status register value
    """
    error_code: int
    error_message: str
    operation_complete: bool
    questionable_status: int
    standard_event_status: int

# ============================================================================
# MAIN POWER SUPPLY CLASS
# ============================================================================

class KeithleyPowerSupply:
    """
    Main class for controlling Keithley 2230 series power supplies.

    This class implements a comprehensive SCPI command wrapper providing full
    control over all power supply functions including voltage/current control,
    output management, measurements, channel combinations, and status monitoring.

    Attributes:
        _visa_address: VISA resource address (e.g., "USB0::0x05E6::0x2230::INSTR")
        _timeout_ms: Communication timeout in milliseconds
        _is_connected: Connection state flag
        _resource_manager: PyVISA resource manager instance
        _instrument: PyVISA instrument resource instance
        _logger: Logging instance for this object
        max_channels: Maximum number of channels (model-dependent)
        max_voltage: Maximum voltage rating (model-dependent)
        max_current: Maximum current rating (model-dependent)
        model: Instrument model identifier
    """

    def __init__(self, visa_address: str, timeout_ms: int = 10000):
        """
        Initialize the Keithley power supply controller.

        Args:
            visa_address: VISA resource address string
            timeout_ms: Communication timeout in milliseconds (default: 10000)
        """
        # VISA communication parameters
        self._visa_address = visa_address
        self._timeout_ms = timeout_ms
        self._is_connected = False

        # VISA resource objects
        self._resource_manager: Optional[pyvisa.ResourceManager] = None
        self._instrument: Any = None  # PyVISA resource object

        # Logging configuration
        self._logger = logging.getLogger(f'{self.__class__.__name__}.{id(self)}')

        # Instrument specifications (updated after connection based on model)
        self.max_channels = 3      # Default: 3 channels
        self.max_voltage = 30.0    # Default: 30V max
        self.max_current = 3.0     # Default: 3A max
        self.model = "Unknown"     # Will be populated from *IDN?

        # Timing parameters for command execution (empirically determined)
        self._voltage_settling_time = 0.5    # Time for voltage to settle after setting
        self._current_settling_time = 0.5    # Time for current to settle after setting
        self._output_enable_time = 0.7       # Time for output to stabilize after enable
        self._reset_time = 3.0               # Time required for instrument reset
        self._channel_switch_time = 0.2      # Time to switch between channels
        self._measurement_time = 0.5         # Time between measurements

        # Valid parameter ranges (updated based on model after connection)
        self._valid_voltage_range = (0.0, 30.0)    # Min/max voltage in volts
        self._valid_current_range = (0.001, 3.0)   # Min/max current in amperes
        self._valid_ovp_range = (1.0, 35.0)        # Min/max OVP level in volts

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    @property
    def is_connected(self) -> bool:
        """
        Check if the instrument is currently connected.

        Returns:
            True if connected and ready, False otherwise
        """
        return self._is_connected and self._instrument is not None

    @property
    def visa_address(self) -> str:
        """
        Get the VISA address of the instrument.

        Returns:
            VISA resource address string
        """
        return self._visa_address

    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================

    def connect(self) -> bool:
        """
        Establish connection to the Keithley power supply.

        This method performs the following operations:
        1. Creates VISA resource manager
        2. Opens connection to the instrument
        3. Configures communication parameters
        4. Queries instrument identification
        5. Configures model-specific parameters
        6. Clears status and synchronizes

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self._logger.info("Attempting to connect to Keithley power supply...")

            # Create VISA resource manager instance
            self._resource_manager = pyvisa.ResourceManager()
            self._logger.info("VISA resource manager created successfully")

            # Open connection to the instrument using VISA address
            self._instrument = self._resource_manager.open_resource(self._visa_address)
            self._logger.info(f"Opened connection to {self._visa_address}")

            # Configure communication parameters
            self._instrument.timeout = self._timeout_ms  # Set timeout for operations
            self._instrument.read_termination = '\n'     # Set read terminator to newline
            self._instrument.write_termination = '\n'    # Set write terminator to newline

            # Query instrument identification using IEEE 488.2 common command
            identification = self._instrument.query("*IDN?")
            self._logger.info(f"Instrument identification: {identification.strip()}")

            # Configure model-specific parameters based on identification string
            self._configure_model_parameters(identification)

            # Clear status registers and error queue using IEEE 488.2 common command
            self._instrument.write("*CLS")
            time.sleep(self._reset_time)  # Allow time for status clearing

            # Wait for operation complete (synchronization point)
            self._instrument.query("*OPC?")

            # Update connection state
            self._is_connected = True
            self._logger.info(f"Successfully connected to Keithley {self.model}")

            return True

        except Exception as e:
            self._logger.error(f"Connection failed: {e}")

            # Clean up resources on failure
            try:
                if self._instrument:
                    self._instrument.close()
                if self._resource_manager:
                    self._resource_manager.close()
            except Exception:
                pass  # Ignore cleanup errors

            # Reset connection state
            self._instrument = None
            self._resource_manager = None
            self._is_connected = False

            return False

    def _configure_model_parameters(self, identification: str):
        """
        Configure model-specific parameters based on identification string.

        Parses the *IDN? response and sets appropriate voltage/current limits
        and channel count based on the detected model.

        Args:
            identification: Response from *IDN? query
        """
        # Parse identification string: "Manufacturer,Model,Serial,Firmware"
        parts = identification.strip().split(',')
        manufacturer = parts[0] if len(parts) > 0 else ""
        model = parts[1] if len(parts) > 1 else ""

        # Validate manufacturer
        if "KEITHLEY" not in manufacturer.upper() and "TEKTRONIX" not in manufacturer.upper():
            self._logger.warning(f"Unexpected manufacturer: {manufacturer}")

        # Configure parameters based on model number
        if "2230" in model:
            # Keithley 2230-30-3: 3 channels, 30V, 3A
            self.max_channels = 3
            self.max_voltage = 30.0
            self.max_current = 3.0
            self.model = "2230-30-3"

        elif "2231" in model:
            # Keithley 2231A-30-3: 3 channels, 30V, 3A (enhanced model)
            self.max_channels = 3
            self.max_voltage = 30.0
            self.max_current = 3.0
            self.model = "2231A-30-3"

        elif "2280S" in model:
            # Keithley 2280S: 1 channel, 72V, 120A (high power model)
            self.max_channels = 1
            self.max_voltage = 72.0
            self.max_current = 120.0
            self.model = "2280S"

        else:
            # Unknown model - use conservative defaults
            self.max_channels = 3
            self.max_voltage = 30.0
            self.max_current = 3.0
            self.model = model.strip()
            self._logger.warning(f"Unknown model {model}, using defaults")

        # Update valid parameter ranges based on model specifications
        self._valid_voltage_range = (0.0, self.max_voltage)
        self._valid_current_range = (0.001, self.max_current)
        self._valid_ovp_range = (1.0, self.max_voltage + 5.0)  # OVP can exceed max voltage

        self._logger.info(
            f"Configured model {self.model}: "
            f"{self.max_channels} channels, "
            f"{self.max_voltage}V/{self.max_current}A max"
        )

    def disconnect(self):
        """
        Safely disconnect from the power supply.

        This method performs the following operations:
        1. Disables all outputs (safety measure)
        2. Closes VISA instrument connection
        3. Closes VISA resource manager
        4. Resets connection state
        """
        try:
            if self._is_connected and self._instrument:
                # Safety measure: disable all outputs before disconnecting
                try:
                    self.disable_all_outputs()
                    time.sleep(0.5)  # Allow outputs to fully disable
                except Exception as e:
                    self._logger.warning(f"Could not disable outputs during disconnect: {e}")

                # Close instrument connection
                self._instrument.close()
                self._logger.info("Instrument connection closed")

            # Close resource manager
            if self._resource_manager:
                self._resource_manager.close()
                self._logger.info("VISA resource manager closed")

        except Exception as e:
            self._logger.error(f"Error during disconnection: {e}")

        finally:
            # Reset all connection-related attributes
            self._instrument = None
            self._resource_manager = None
            self._is_connected = False
            self._logger.info("Disconnection completed")

    # ========================================================================
    # INSTRUMENT INFORMATION
    # ========================================================================

    def get_instrument_info(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve comprehensive instrument identification information.

        Queries the instrument's identification string and returns a dictionary
        with parsed components including manufacturer, model, serial number, 
        firmware version, and capabilities.

        Returns:
            Dictionary with instrument information, or None if query fails
            Keys include: manufacturer, model, serial_number, firmware_version,
            max_channels, max_voltage, max_current, visa_address, identification
        """
        if not self.is_connected:
            self._logger.error("Cannot get info: not connected")
            return None

        try:
            # Query IEEE 488.2 identification string
            idn = self._instrument.query("*IDN?").strip()

            # Parse identification string components
            parts = idn.split(',')

            # Build and return information dictionary
            return {
                'manufacturer': parts[0] if len(parts) > 0 else 'Unknown',
                'model': parts[1] if len(parts) > 1 else 'Unknown',
                'serial_number': parts[2] if len(parts) > 2 else 'Unknown',
                'firmware_version': parts[3] if len(parts) > 3 else 'Unknown',
                'max_channels': self.max_channels,
                'max_voltage': self.max_voltage,
                'max_current': self.max_current,
                'visa_address': self._visa_address,
                'identification': idn
            }

        except Exception as e:
            self._logger.error(f"Failed to get instrument info: {e}")
            return None

    def get_scpi_version(self) -> Optional[str]:
        """
        Query the SCPI version supported by the instrument.

        SCPI Command: :SYSTem:VERSion?

        Returns:
            SCPI version string (e.g., "1991.0"), or None if query fails
        """
        if not self.is_connected:
            self._logger.error("Cannot query SCPI version: not connected")
            return None

        try:
            version = self._instrument.query(":SYSTem:VERSion?").strip()
            self._logger.info(f"SCPI version: {version}")
            return version
        except Exception as e:
            self._logger.error(f"Failed to query SCPI version: {e}")
            return None

    # ========================================================================
    # CHANNEL SELECTION
    # ========================================================================

    def select_channel(self, channel: int) -> bool:
        """
        Select the active channel for subsequent operations.

        SCPI Command: :INSTrument:SELect CH{channel}

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            True if channel selected successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot select channel: not connected")
            return False

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)
            self._logger.debug(f"Selected channel {channel}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to select channel {channel}: {e}")
            return False

    def select_channel_by_number(self, channel: int) -> bool:
        """
        Select channel using numeric identifier (alternate command).

        SCPI Command: :INSTrument:NSELect {channel}

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            True if channel selected successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot select channel: not connected")
            return False

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False

        try:
            self._instrument.write(f":INSTrument:NSELect {channel}")
            time.sleep(self._channel_switch_time)
            self._logger.debug(f"Selected channel {channel} (by number)")
            return True
        except Exception as e:
            self._logger.error(f"Failed to select channel {channel}: {e}")
            return False

    def get_selected_channel(self) -> Optional[int]:
        """
        Query the currently selected channel.

        SCPI Command: :INSTrument:SELect?

        Returns:
            Channel number (1, 2, or 3), or None if query fails
        """
        if not self.is_connected:
            self._logger.error("Cannot query selected channel: not connected")
            return None

        try:
            response = self._instrument.query(":INSTrument:SELect?").strip()
            # Response format: "CH1", "CH2", or "CH3"
            channel_num = int(response.replace("CH", ""))
            self._logger.debug(f"Currently selected channel: {channel_num}")
            return channel_num
        except Exception as e:
            self._logger.error(f"Failed to query selected channel: {e}")
            return None

    # ========================================================================
    # VOLTAGE CONTROL
    # ========================================================================

    def set_voltage(self, channel: int, voltage: float) -> bool:
        """
        Set the output voltage for a specific channel.

        SCPI Command: :SOURce:VOLTage:LEVel:IMMediate:AMPLitude {voltage}

        Args:
            channel: Channel number (1, 2, or 3)
            voltage: Desired voltage in volts

        Returns:
            True if voltage set successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set voltage: not connected")
            return False

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False

        if not (self._valid_voltage_range[0] <= voltage <= self._valid_voltage_range[1]):
            self._logger.error(
                f"Voltage {voltage}V out of range {self._valid_voltage_range}"
            )
            return False

        try:
            # Select the target channel
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            # Set voltage using full SCPI command form
            self._instrument.write(f":SOURce:VOLTage {voltage}")
            time.sleep(self._voltage_settling_time)

            # Verify the set voltage
            actual_voltage = float(self._instrument.query(":SOURce:VOLTage?"))
            self._logger.info(
                f"CH{channel} voltage set to {actual_voltage:.6f}V "
                f"(requested: {voltage:.6f}V)"
            )
            return True

        except Exception as e:
            self._logger.error(f"Failed to set voltage on CH{channel}: {e}")
            return False

    def get_voltage(self, channel: int) -> Optional[float]:
        """
        Query the set voltage for a specific channel.

        SCPI Command: :SOURce:VOLTage:LEVel:IMMediate:AMPLitude?

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            Set voltage in volts, or None if query fails
        """
        if not self.is_connected:
            self._logger.error("Cannot query voltage: not connected")
            return None

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return None

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            voltage = float(self._instrument.query(":SOURce:VOLTage?"))
            self._logger.debug(f"CH{channel} set voltage: {voltage:.6f}V")
            return voltage

        except Exception as e:
            self._logger.error(f"Failed to query voltage on CH{channel}: {e}")
            return None

    def set_voltage_step(self, channel: int, step_size: float) -> bool:
        """
        Configure the voltage step increment for up/down commands.

        SCPI Command: :SOURce:VOLTage:LEVel:IMMediate:STEP:INCRement {step_size}

        Args:
            channel: Channel number (1, 2, or 3)
            step_size: Voltage step size in volts

        Returns:
            True if step size set successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set voltage step: not connected")
            return False

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            self._instrument.write(
                f":SOURce:VOLTage:LEVel:IMMediate:STEP:INCRement {step_size}"
            )
            self._logger.info(f"CH{channel} voltage step set to {step_size}V")
            return True

        except Exception as e:
            self._logger.error(f"Failed to set voltage step on CH{channel}: {e}")
            return False

    def voltage_step_up(self, channel: int) -> bool:
        """
        Increase voltage by one step increment.

        SCPI Command: :SOURce:VOLTage:LEVel:UP:IMMediate:AMPLitude

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            True if voltage increased successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot step up voltage: not connected")
            return False

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            self._instrument.write(":SOURce:VOLTage:LEVel:UP:IMMediate:AMPLitude")
            time.sleep(self._voltage_settling_time)

            new_voltage = float(self._instrument.query(":SOURce:VOLTage?"))
            self._logger.info(f"CH{channel} voltage stepped up to {new_voltage:.6f}V")
            return True

        except Exception as e:
            self._logger.error(f"Failed to step up voltage on CH{channel}: {e}")
            return False

    def voltage_step_down(self, channel: int) -> bool:
        """
        Decrease voltage by one step increment.

        SCPI Command: :SOURce:VOLTage:LEVel:DOWN:IMMediate:AMPLitude

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            True if voltage decreased successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot step down voltage: not connected")
            return False

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            self._instrument.write(":SOURce:VOLTage:LEVel:DOWN:IMMediate:AMPLitude")
            time.sleep(self._voltage_settling_time)

            new_voltage = float(self._instrument.query(":SOURce:VOLTage?"))
            self._logger.info(f"CH{channel} voltage stepped down to {new_voltage:.6f}V")
            return True

        except Exception as e:
            self._logger.error(f"Failed to step down voltage on CH{channel}: {e}")
            return False

    # ========================================================================
    # CURRENT CONTROL
    # ========================================================================

    def set_current_limit(self, channel: int, current_limit: float) -> bool:
        """
        Set the current limit for a specific channel.

        SCPI Command: :SOURce:CURRent:LEVel:IMMediate:AMPLitude {current_limit}

        Args:
            channel: Channel number (1, 2, or 3)
            current_limit: Desired current limit in amperes

        Returns:
            True if current limit set successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set current limit: not connected")
            return False

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False

        if not (self._valid_current_range[0] <= current_limit <= self._valid_current_range[1]):
            self._logger.error(
                f"Current limit {current_limit}A out of range {self._valid_current_range}"
            )
            return False

        try:
            # Select the target channel
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            # Set current limit using full SCPI command form
            self._instrument.write(f":SOURce:CURRent {current_limit}")
            time.sleep(self._current_settling_time)

            # Verify the set current limit
            actual_current = float(self._instrument.query(":SOURce:CURRent?"))
            self._logger.info(
                f"CH{channel} current limit set to {actual_current:.6f}A "
                f"(requested: {current_limit:.6f}A)"
            )
            return True

        except Exception as e:
            self._logger.error(f"Failed to set current limit on CH{channel}: {e}")
            return False

    def get_current_limit(self, channel: int) -> Optional[float]:
        """
        Query the set current limit for a specific channel.

        SCPI Command: :SOURce:CURRent:LEVel:IMMediate:AMPLitude?

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            Set current limit in amperes, or None if query fails
        """
        if not self.is_connected:
            self._logger.error("Cannot query current limit: not connected")
            return None

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return None

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            current = float(self._instrument.query(":SOURce:CURRent?"))
            self._logger.debug(f"CH{channel} set current limit: {current:.6f}A")
            return current

        except Exception as e:
            self._logger.error(f"Failed to query current limit on CH{channel}: {e}")
            return None

    def set_current_step(self, channel: int, step_size: float) -> bool:
        """
        Configure the current step increment for up/down commands.

        SCPI Command: :SOURce:CURRent:LEVel:IMMediate:STEP:INCRement {step_size}

        Args:
            channel: Channel number (1, 2, or 3)
            step_size: Current step size in amperes

        Returns:
            True if step size set successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set current step: not connected")
            return False

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            self._instrument.write(
                f":SOURce:CURRent:LEVel:IMMediate:STEP:INCRement {step_size}"
            )
            self._logger.info(f"CH{channel} current step set to {step_size}A")
            return True

        except Exception as e:
            self._logger.error(f"Failed to set current step on CH{channel}: {e}")
            return False

    def current_step_up(self, channel: int) -> bool:
        """
        Increase current limit by one step increment.

        SCPI Command: :SOURce:CURRent:LEVel:UP:IMMediate:AMPLitude

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            True if current increased successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot step up current: not connected")
            return False

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            self._instrument.write(":SOURce:CURRent:LEVel:UP:IMMediate:AMPLitude")
            time.sleep(self._current_settling_time)

            new_current = float(self._instrument.query(":SOURce:CURRent?"))
            self._logger.info(f"CH{channel} current stepped up to {new_current:.6f}A")
            return True

        except Exception as e:
            self._logger.error(f"Failed to step up current on CH{channel}: {e}")
            return False

    def current_step_down(self, channel: int) -> bool:
        """
        Decrease current limit by one step increment.

        SCPI Command: :SOURce:CURRent:LEVel:DOWN:IMMediate:AMPLitude

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            True if current decreased successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot step down current: not connected")
            return False

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            self._instrument.write(":SOURce:CURRent:LEVel:DOWN:IMMediate:AMPLitude")
            time.sleep(self._current_settling_time)

            new_current = float(self._instrument.query(":SOURce:CURRent?"))
            self._logger.info(f"CH{channel} current stepped down to {new_current:.6f}V")
            return True

        except Exception as e:
            self._logger.error(f"Failed to step down current on CH{channel}: {e}")
            return False
    
    # ========================================================================
    # VOLTAGE/CURRENT LIMIT CONTROL
    # ========================================================================

    def set_voltage_limit(self, channel: int, voltage_limit: float) -> bool:
        """
        Set the maximum voltage limit for a channel.

        SCPI Command: :SOURce:VOLTage:LIMit:LEVel {voltage_limit}

        Args:
            channel: Channel number (1, 2, or 3)
            voltage_limit: Maximum voltage limit in volts

        Returns:
            True if limit set successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set voltage limit: not connected")
            return False

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            self._instrument.write(f":SOURce:VOLTage:LIMit:LEVel {voltage_limit}")
            self._logger.info(f"CH{channel} voltage limit set to {voltage_limit}V")
            return True

        except Exception as e:
            self._logger.error(f"Failed to set voltage limit on CH{channel}: {e}")
            return False

    def get_voltage_limit(self, channel: int) -> Optional[float]:
        """
        Query the voltage limit for a specific channel.

        SCPI Command: :SOURce:VOLTage:LIMit:LEVel?

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            Voltage limit in volts, or None if query fails
        """
        if not self.is_connected:
            self._logger.error("Cannot query voltage limit: not connected")
            return None

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return None

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            limit = float(self._instrument.query(":SOURce:VOLTage:LIMit:LEVel?"))
            self._logger.debug(f"CH{channel} voltage limit: {limit:.6f}V")
            return limit

        except Exception as e:
            self._logger.error(f"Failed to query voltage limit on CH{channel}: {e}")
            return None

    def enable_voltage_limit(self, channel: int, enable: bool = True) -> bool:
        """
        Enable or disable the voltage limit function.

        SCPI Command: :SOURce:VOLTage:LIMit:STATe {ON|OFF}

        Args:
            channel: Channel number (1, 2, or 3)
            enable: True to enable limit, False to disable

        Returns:
            True if state set successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set voltage limit state: not connected")
            return False

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            state = "ON" if enable else "OFF"
            self._instrument.write(f":SOURce:VOLTage:LIMit:STATe {state}")
            self._logger.info(f"CH{channel} voltage limit {state}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to set voltage limit state on CH{channel}: {e}")
            return False

    def get_voltage_limit_state(self, channel: int) -> Optional[bool]:
        """
        Query the voltage limit enable state.

        SCPI Command: :SOURce:VOLTage:LIMit:STATe?

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            True if enabled, False if disabled, None if query fails
        """
        if not self.is_connected:
            self._logger.error("Cannot query voltage limit state: not connected")
            return None

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return None

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            state = self._instrument.query(":SOURce:VOLTage:LIMit:STATe?").strip().upper()
            enabled = state in ("1", "ON")
            self._logger.debug(f"CH{channel} voltage limit state: {enabled}")
            return enabled

        except Exception as e:
            self._logger.error(f"Failed to query voltage limit state on CH{channel}: {e}")
            return None

    # ========================================================================
    # PROTECTION CONTROL
    # ========================================================================

    def set_ovp_level(self, channel: int, ovp_level: float) -> bool:
        """
        Set the over-voltage protection (OVP) threshold.

        SCPI Command: :SOURce:VOLTage:PROTection {ovp_level}

        Args:
            channel: Channel number (1, 2, or 3)
            ovp_level: OVP threshold in volts

        Returns:
            True if OVP level set successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set OVP level: not connected")
            return False

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False

        if not (self._valid_ovp_range[0] <= ovp_level <= self._valid_ovp_range[1]):
            self._logger.error(
                f"OVP level {ovp_level}V out of range {self._valid_ovp_range}"
            )
            return False

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            self._instrument.write(f":SOURce:VOLTage:PROTection {ovp_level}")
            time.sleep(0.3)

            self._logger.info(f"CH{channel} OVP level set to {ovp_level}V")
            return True

        except Exception as e:
            self._logger.error(f"Failed to set OVP level on CH{channel}: {e}")
            return False

    def get_ovp_level(self, channel: int) -> Optional[float]:
        """
        Query the over-voltage protection (OVP) threshold.

        SCPI Command: :SOURce:VOLTage:PROTection?

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            OVP threshold in volts, or None if query fails
        """
        if not self.is_connected:
            self._logger.error("Cannot query OVP level: not connected")
            return None

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return None

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            ovp = float(self._instrument.query(":SOURce:VOLTage:PROTection?"))
            self._logger.debug(f"CH{channel} OVP level: {ovp:.6f}V")
            return ovp

        except Exception as e:
            self._logger.error(f"Failed to query OVP level on CH{channel}: {e}")
            return None

    def clear_protection(self) -> bool:
        """
        Clear all protection states (OVP, OCP) on all channels.

        SCPI Command: :SOURce:OUTPut:PROTection:CLEar

        Returns:
            True if protection cleared successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot clear protection: not connected")
            return False

        try:
            self._instrument.write(":SOURce:OUTPut:PROTection:CLEar")
            time.sleep(0.5)
            self._logger.info("Protection states cleared on all channels")
            return True

        except Exception as e:
            self._logger.error(f"Failed to clear protection: {e}")
            return False

    # ========================================================================
    # OUTPUT CONTROL
    # ========================================================================

    def enable_channel_output(self, channel: int) -> bool:
        """
        Enable the output on a specific channel.

        SCPI Command: :OUTPut ON

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            True if output enabled successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot enable output: not connected")
            return False

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False

        try:
            self._logger.info(f"Enabling output on CH{channel}")

            # Select channel and enable output
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            self._instrument.write(":OUTPut ON")
            time.sleep(self._output_enable_time)

            # Verify output state
            state = self._instrument.query(":OUTPut?").strip().upper()

            if state in ("1", "ON"):
                self._logger.info(f"CH{channel} output enabled")
                return True
            else:
                self._logger.error(f"CH{channel} output enable failed; state='{state}'")
                return False

        except Exception as e:
            self._logger.error(f"Enable output failed on CH{channel}: {e}")
            return False

    def disable_channel_output(self, channel: int) -> bool:
        """
        Disable the output on a specific channel.

        SCPI Command: :OUTPut OFF

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            True if output disabled successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot disable output: not connected")
            return False

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False

        try:
            self._logger.info(f"Disabling output on CH{channel}")

            # Select channel and disable output
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            self._instrument.write(":OUTPut OFF")
            time.sleep(0.5)

            # Verify output state
            state = self._instrument.query(":OUTPut?").strip().upper()

            if state in ("0", "OFF"):
                self._logger.info(f"CH{channel} output disabled")
                return True
            else:
                self._logger.error(f"CH{channel} output disable failed; state='{state}'")
                return False

        except Exception as e:
            self._logger.error(f"Disable output failed on CH{channel}: {e}")
            return False

    def get_output_state(self, channel: int) -> Optional[bool]:
        """
        Query the output enable state of a specific channel.

        SCPI Command: :OUTPut?

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            True if enabled, False if disabled, None if query fails
        """
        if not self.is_connected:
            self._logger.error("Cannot query output state: not connected")
            return None

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return None

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            state = self._instrument.query(":OUTPut?").strip().upper()
            enabled = state in ("1", "ON")
            self._logger.debug(f"CH{channel} output state: {enabled}")
            return enabled

        except Exception as e:
            self._logger.error(f"Failed to query output state on CH{channel}: {e}")
            return None

    def enable_all_outputs(self) -> bool:
        """
        Enable outputs on all channels simultaneously.

        SCPI Command: :SOURce:OUTPut:STATe:ALL ON

        Returns:
            True if all outputs enabled successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot enable outputs: not connected")
            return False

        try:
            self._instrument.write(":SOURce:OUTPut:STATe:ALL ON")
            time.sleep(self._output_enable_time)
            self._logger.info("All outputs enabled")
            return True

        except Exception as e:
            self._logger.error(f"Failed to enable all outputs: {e}")
            return False

    def disable_all_outputs(self) -> bool:
        """
        Disable outputs on all channels simultaneously.

        SCPI Command: :SOURce:OUTPut:STATe:ALL OFF

        Returns:
            True if all outputs disabled successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot disable outputs: not connected")
            return False

        try:
            self._instrument.write(":SOURce:OUTPut:STATe:ALL OFF")
            time.sleep(0.5)
            self._logger.info("All outputs disabled")
            return True

        except Exception as e:
            self._logger.error(f"Failed to disable all outputs: {e}")
            return False

    def set_channel_output_state(self, channel: int, enable: bool) -> bool:
        """
        Set the output state using channel-specific command.

        SCPI Command: :SOURce:CHANnel:OUTPut:STATe {ON|OFF}

        Args:
            channel: Channel number (1, 2, or 3)
            enable: True to enable, False to disable

        Returns:
            True if state set successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set channel output state: not connected")
            return False

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            state = "ON" if enable else "OFF"
            self._instrument.write(f":SOURce:CHANnel:OUTPut:STATe {state}")
            time.sleep(self._output_enable_time if enable else 0.5)

            self._logger.info(f"CH{channel} output state set to {state}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to set channel output state on CH{channel}: {e}")
            return False

    # ========================================================================
    # TIMER CONTROL
    # ========================================================================

    def set_output_timer_delay(self, channel: int, delay_seconds: float) -> bool:
        """
        Set the output timer delay for automatic output control.

        SCPI Command: :SOURce:OUTPut:TIMer:DELay {delay}

        Args:
            channel: Channel number (1, 2, or 3)
            delay_seconds: Delay time in seconds (0.1 to 99999.9)

        Returns:
            True if delay set successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set timer delay: not connected")
            return False

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False

        if not (0.1 <= delay_seconds <= 99999.9):
            self._logger.error(f"Timer delay {delay_seconds}s out of valid range [0.1, 99999.9]")
            return False

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            self._instrument.write(f":SOURce:OUTPut:TIMer:DELay {delay_seconds}")
            self._logger.info(f"CH{channel} timer delay set to {delay_seconds}s")
            return True

        except Exception as e:
            self._logger.error(f"Failed to set timer delay on CH{channel}: {e}")
            return False

    def get_output_timer_delay(self, channel: int) -> Optional[float]:
        """
        Query the output timer delay setting.

        SCPI Command: :SOURce:OUTPut:TIMer:DELay?

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            Timer delay in seconds, or None if query fails
        """
        if not self.is_connected:
            self._logger.error("Cannot query timer delay: not connected")
            return None

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return None

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            delay = float(self._instrument.query(":SOURce:OUTPut:TIMer:DELay?"))
            self._logger.debug(f"CH{channel} timer delay: {delay}s")
            return delay

        except Exception as e:
            self._logger.error(f"Failed to query timer delay on CH{channel}: {e}")
            return None

    def enable_output_timer(self, channel: int, enable: bool = True) -> bool:
        """
        Enable or disable the output timer function.

        SCPI Command: :SOURce:OUTPut:TIMer:STATe {ON|OFF}

        Args:
            channel: Channel number (1, 2, or 3)
            enable: True to enable timer, False to disable

        Returns:
            True if timer state set successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set timer state: not connected")
            return False

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            state = "ON" if enable else "OFF"
            self._instrument.write(f":SOURce:OUTPut:TIMer:STATe {state}")
            self._logger.info(f"CH{channel} timer {state}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to set timer state on CH{channel}: {e}")
            return False

    def get_output_timer_state(self, channel: int) -> Optional[bool]:
        """
        Query the output timer enable state.

        SCPI Command: :SOURce:OUTPut:TIMer:STATe?

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            True if timer enabled, False if disabled, None if query fails
        """
        if not self.is_connected:
            self._logger.error("Cannot query timer state: not connected")
            return None

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return None

        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            state = self._instrument.query(":SOURce:OUTPut:TIMer:STATe?").strip().upper()
            enabled = state in ("1", "ON")
            self._logger.debug(f"CH{channel} timer state: {enabled}")
            return enabled

        except Exception as e:
            self._logger.error(f"Failed to query timer state on CH{channel}: {e}")
            return None

    # ========================================================================
    # CHANNEL CONFIGURATION
    # ========================================================================

    def configure_channel(self, channel: int, voltage: float, current_limit: float, 
                         ovp_level: float, enable_output: bool = False) -> bool:
        """
        Configure all parameters for a channel in a single operation.

        This method provides a convenient way to set voltage, current limit,
        and OVP level, and optionally enable the output, all in one call.

        Args:
            channel: Channel number (1, 2, or 3)
            voltage: Desired voltage in volts
            current_limit: Current limit in amperes
            ovp_level: Over-voltage protection threshold in volts
            enable_output: Whether to enable output after configuration

        Returns:
            True if configuration successful, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot configure channel: not connected")
            return False

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False

        # Validate parameters
        if not (self._valid_voltage_range[0] <= voltage <= self._valid_voltage_range[1]):
            self._logger.error(f"Voltage {voltage}V out of range {self._valid_voltage_range}")
            return False

        if not (self._valid_current_range[0] <= current_limit <= self._valid_current_range[1]):
            self._logger.error(f"Current limit {current_limit}A out of range {self._valid_current_range}")
            return False

        # Ensure OVP level is higher than voltage setting
        if ovp_level <= voltage:
            self._logger.warning(
                f"OVP {ovp_level}V must be > voltage {voltage}V; "
                f"adjusting to {voltage+1.0}V"
            )
            ovp_level = voltage + 1.0

        try:
            # Select the target channel
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._channel_switch_time)

            # Set voltage
            self._instrument.write(f":SOURce:VOLTage {voltage}")
            time.sleep(self._voltage_settling_time)

            # Set current limit
            self._instrument.write(f":SOURce:CURRent {current_limit}")
            time.sleep(self._current_settling_time)

            # Set OVP level
            self._instrument.write(f":SOURce:VOLTage:PROTection {ovp_level}")
            time.sleep(0.3)

            # Enable output if requested
            if enable_output:
                self._instrument.write(":OUTPut ON")
                time.sleep(self._output_enable_time)

            # Verify configuration
            actual_voltage = float(self._instrument.query(":SOURce:VOLTage?"))
            actual_current = float(self._instrument.query(":SOURce:CURRent?"))

            self._logger.info(
                f"CH{channel} configured: {actual_voltage:.6f}V, "
                f"{actual_current:.6f}A limit, "
                f"Output: {'Enabled' if enable_output else 'Disabled'}"
            )
            return True

        except Exception as e:
            self._logger.error(f"Failed to configure channel {channel}: {e}")
            return False

    def apply_channel_settings(self, channel: int, voltage: float, current_limit: float) -> bool:
        """
        Apply voltage and current settings using single APPLY command.

        SCPI Command: :SOURce:APPLy CH{channel}, {voltage}, {current}

        Args:
            channel: Channel number (1, 2, or 3)
            voltage: Desired voltage in volts
            current_limit: Current limit in amperes

        Returns:
            True if settings applied successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot apply settings: not connected")
            return False

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False

        if not (self._valid_voltage_range[0] <= voltage <= self._valid_voltage_range[1]):
            self._logger.error(f"Voltage {voltage}V out of range {self._valid_voltage_range}")
            return False

        if not (self._valid_current_range[0] <= current_limit <= self._valid_current_range[1]):
            self._logger.error(f"Current limit {current_limit}A out of range {self._valid_current_range}")
            return False

        try:
            # Use the APPLY command for simultaneous voltage/current setting
            self._instrument.write(f":SOURce:APPLy CH{channel}, {voltage}, {current_limit}")
            time.sleep(max(self._voltage_settling_time, self._current_settling_time))

            self._logger.info(f"CH{channel} settings applied: {voltage}V, {current_limit}A")
            return True

        except Exception as e:
            self._logger.error(f"Failed to apply settings on CH{channel}: {e}")
            return False
    
    # ========================================================================
    # MEASUREMENT FUNCTIONS
    # ========================================================================

    def measure_voltage(self, channel: int) -> Optional[float]:
        """
        Perform a new voltage measurement on the specified channel.

        SCPI Command: :MEASure:SCALar:VOLTage:DC? CH{channel}

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            Measured voltage in volts, or None if measurement fails
        """
        if not self.is_connected:
            self._logger.error("Cannot measure voltage: not connected")
            return None

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return None

        try:
            # Initiate and execute new measurement
            voltage_str = self._instrument.query(f":MEASure:SCALar:VOLTage:DC? CH{channel}").strip()
            voltage = self._extract_first_float(voltage_str)

            self._logger.debug(f"CH{channel} measured voltage: {voltage:.4f}V")
            return voltage

        except Exception as e:
            self._logger.error(f"Failed to measure voltage on CH{channel}: {e}")
            return None

    def measure_current(self, channel: int) -> Optional[float]:
        """
        Perform a new current measurement on the specified channel.

        SCPI Command: :MEASure:SCALar:CURRent:DC? CH{channel}

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            Measured current in amperes, or None if measurement fails
        """
        if not self.is_connected:
            self._logger.error("Cannot measure current: not connected")
            return None

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return None

        try:
            # Initiate and execute new measurement
            current_str = self._instrument.query(f":MEASure:SCALar:CURRent:DC? CH{channel}").strip()
            current = self._extract_first_float(current_str)

            self._logger.debug(f"CH{channel} measured current: {current:.4f}A")
            return current

        except Exception as e:
            self._logger.error(f"Failed to measure current on CH{channel}: {e}")
            return None

    def measure_power(self, channel: int) -> Optional[float]:
        """
        Perform a new power measurement on the specified channel.

        Power is calculated as voltage × current by the instrument.

        SCPI Command: :MEASure:SCALar:POWer:DC? CH{channel}

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            Measured power in watts, or None if measurement fails
        """
        if not self.is_connected:
            self._logger.error("Cannot measure power: not connected")
            return None

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return None

        try:
            # Initiate and execute new measurement
            power_str = self._instrument.query(f":MEASure:SCALar:POWer:DC? CH{channel}").strip()
            power = self._extract_first_float(power_str)

            self._logger.debug(f"CH{channel} measured power: {power:.4f}W")
            return power

        except Exception as e:
            self._logger.error(f"Failed to measure power on CH{channel}: {e}")
            return None

    def fetch_voltage(self, channel: int) -> Optional[float]:
        """
        Fetch the last voltage measurement without initiating a new one.

        SCPI Command: :FETCh:SCALar:VOLTage:DC? CH{channel}

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            Last measured voltage in volts, or None if fetch fails
        """
        if not self.is_connected:
            self._logger.error("Cannot fetch voltage: not connected")
            return None

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return None

        try:
            voltage_str = self._instrument.query(f":FETCh:SCALar:VOLTage:DC? CH{channel}").strip()
            voltage = self._extract_first_float(voltage_str)

            self._logger.debug(f"CH{channel} fetched voltage: {voltage:.4f}V")
            return voltage

        except Exception as e:
            self._logger.error(f"Failed to fetch voltage on CH{channel}: {e}")
            return None

    def fetch_current(self, channel: int) -> Optional[float]:
        """
        Fetch the last current measurement without initiating a new one.

        SCPI Command: :FETCh:SCALar:CURRent:DC? CH{channel}

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            Last measured current in amperes, or None if fetch fails
        """
        if not self.is_connected:
            self._logger.error("Cannot fetch current: not connected")
            return None

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return None

        try:
            current_str = self._instrument.query(f":FETCh:SCALar:CURRent:DC? CH{channel}").strip()
            current = self._extract_first_float(current_str)

            self._logger.debug(f"CH{channel} fetched current: {current:.4f}A")
            return current

        except Exception as e:
            self._logger.error(f"Failed to fetch current on CH{channel}: {e}")
            return None

    def fetch_power(self, channel: int) -> Optional[float]:
        """
        Fetch the last power measurement without initiating a new one.

        SCPI Command: :FETCh:SCALar:POWer:DC? CH{channel}

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            Last measured power in watts, or None if fetch fails
        """
        if not self.is_connected:
            self._logger.error("Cannot fetch power: not connected")
            return None

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return None

        try:
            power_str = self._instrument.query(f":FETCh:SCALar:POWer:DC? CH{channel}").strip()
            power = self._extract_first_float(power_str)

            self._logger.debug(f"CH{channel} fetched power: {power:.4f}W")
            return power

        except Exception as e:
            self._logger.error(f"Failed to fetch power on CH{channel}: {e}")
            return None

    def measure_all_channels(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Measure voltage and current on all channels simultaneously.

        SCPI Commands: 
            :MEASure:SCALar:VOLTage:DC? ALL
            :MEASure:SCALar:CURRent:DC? ALL

        Returns:
            Dictionary with measurements for all channels, or None if measurement fails
            Format: {'CH1': {'voltage': v, 'current': i}, 'CH2': {...}, 'CH3': {...}}
        """
        if not self.is_connected:
            self._logger.error("Cannot measure all channels: not connected")
            return None

        try:
            # Measure voltages on all channels
            voltage_response = self._instrument.query(":MEASure:SCALar:VOLTage:DC? ALL").strip()
            voltages = [float(v) for v in voltage_response.split(',')]

            # Measure currents on all channels
            current_response = self._instrument.query(":MEASure:SCALar:CURRent:DC? ALL").strip()
            currents = [float(i) for i in current_response.split(',')]

            # Build results dictionary
            results = {}
            for ch in range(1, self.max_channels + 1):
                results[f'CH{ch}'] = {
                    'voltage': voltages[ch-1] if ch <= len(voltages) else 0.0,
                    'current': currents[ch-1] if ch <= len(currents) else 0.0,
                    'power': voltages[ch-1] * currents[ch-1] if ch <= len(voltages) else 0.0
                }

            self._logger.debug(f"All channels measured: {results}")
            return results

        except Exception as e:
            self._logger.error(f"Failed to measure all channels: {e}")
            return None

    def measure_channel_output(self, channel: int) -> Optional[Tuple[float, float]]:
        """
        Comprehensive measurement function with enhanced parsing and error handling.

        This is the legacy method maintained for backward compatibility.
        Measures voltage and current, with improved buffer management and parsing.

        Args:
            channel: Channel number (1, 2, or 3)

        Returns:
            Tuple of (voltage, current) in volts and amperes, or None if measurement fails
        """
        if not self.is_connected:
            self._logger.error("Cannot measure: not connected")
            return None

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return None

        # Store original timeout for restoration
        original_timeout = self._instrument.timeout

        try:
            self._logger.info(f"Measuring channel {channel}...")

            # Extend timeout for potentially slow measurements
            self._instrument.timeout = 15000  # 15 seconds

            # Clear device buffer to remove stale data
            try:
                self._instrument.clear()
            except Exception as clear_err:
                # Not all VISA backends support clear()
                self._logger.debug(f"Buffer clear not supported or failed: {clear_err}")

            # Select channel
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(self._measurement_time)

            # Measure voltage
            voltage_str = self._instrument.query(":MEASure:VOLTage?").strip()
            self._logger.info(f"Raw voltage response: '{voltage_str}'")
            time.sleep(self._measurement_time)

            # Measure current
            current_str = self._instrument.query(":MEASure:CURRent?").strip()
            self._logger.info(f"Raw current response: '{current_str}'")

            # Parse measurements using robust extraction
            voltage = self._extract_first_float(voltage_str)
            current = self._extract_first_float(current_str)

            # Check output state and sanitize current if output is off
            try:
                state_str = self._instrument.query(":OUTPut?").strip()
                self._logger.debug(f"Output state: '{state_str}'")

                if state_str in ['0', 'OFF', 'off']:
                    if abs(current) > 0.001:
                        self._logger.warning(
                            f"Output OFF but current={current}A, forcing to 0"
                        )
                        current = 0.0
            except Exception as state_err:
                self._logger.debug(f"Could not check output state: {state_err}")

            # Validate readings against expected ranges
            if voltage < 0 or voltage > (self.max_voltage + 5):
                self._logger.warning(f"Unrealistic voltage: {voltage}V")

            if current < 0 or current > (self.max_current + 2):
                self._logger.warning(f"Unrealistic current: {current}A")

            self._logger.info(f"Channel {channel} final: {voltage:.4f}V, {current:.4f}A")
            return (voltage, current)

        except Exception as e:
            self._logger.error(f"Measurement failed on channel {channel}: {e}")
            import traceback
            self._logger.error(traceback.format_exc())
            return None

        finally:
            # Always restore original timeout
            try:
                self._instrument.timeout = original_timeout
            except Exception as restore_err:
                self._logger.debug(f"Failed to restore timeout: {restore_err}")

    def _extract_first_float(self, s: str) -> float:
        """
        Extract the first numeric value from a string using regex.

        Handles various numeric formats including scientific notation.

        Args:
            s: String potentially containing numeric data

        Returns:
            First extracted float value, or 0.0 if no valid number found
        """
        # Regular expression to match floating point numbers (including scientific notation)
        matches = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', s)

        if matches:
            return float(matches[0])
        else:
            self._logger.warning(f"Could not parse number from '{s}', returning 0.0")
            return 0.0

    # ========================================================================
    # CHANNEL COMBINATION MODES
    # ========================================================================

    def set_tracking_mode(self, channels: str) -> bool:
        """
        Enable tracking mode where selected channels mirror each other's settings.

        SCPI Command: :INSTrument:COMBine:TRACk {channels}

        Args:
            channels: Channel combination string
                     Valid values: "CH1CH2", "CH2CH3", "CH1CH2CH3"

        Returns:
            True if tracking mode enabled successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set tracking mode: not connected")
            return False

        valid_combinations = ["CH1CH2", "CH2CH3", "CH1CH2CH3"]
        if channels not in valid_combinations:
            self._logger.error(f"Invalid channel combination: {channels}")
            return False

        try:
            self._instrument.write(f":INSTrument:COMBine:TRACk {channels}")
            time.sleep(0.5)
            self._logger.info(f"Tracking mode enabled for {channels}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to set tracking mode: {e}")
            return False

    def set_series_mode(self) -> bool:
        """
        Combine CH1 and CH2 in series (voltages add, currents remain same).

        SCPI Command: :INSTrument:COMBine:SERies

        Returns:
            True if series mode enabled successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set series mode: not connected")
            return False

        try:
            self._instrument.write(":INSTrument:COMBine:SERies")
            time.sleep(0.5)
            self._logger.info("Series mode enabled for CH1 and CH2")
            return True

        except Exception as e:
            self._logger.error(f"Failed to set series mode: {e}")
            return False

    def set_parallel_mode(self, channels: str) -> bool:
        """
        Combine channels in parallel (currents add, voltages remain same).

        SCPI Command: :INSTrument:COMBine:PARallel {channels}

        Args:
            channels: Channel combination string
                     Valid values: "CH1CH2", "CH2CH3", "CH1CH2CH3"

        Returns:
            True if parallel mode enabled successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set parallel mode: not connected")
            return False

        valid_combinations = ["CH1CH2", "CH2CH3", "CH1CH2CH3"]
        if channels not in valid_combinations:
            self._logger.error(f"Invalid channel combination: {channels}")
            return False

        try:
            self._instrument.write(f":INSTrument:COMBine:PARallel {channels}")
            time.sleep(0.5)
            self._logger.info(f"Parallel mode enabled for {channels}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to set parallel mode: {e}")
            return False

    def disable_channel_combination(self) -> bool:
        """
        Disable all channel combination modes (tracking, series, parallel).

        SCPI Command: :INSTrument:COMBine:OFF

        Returns:
            True if combination mode disabled successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot disable combination mode: not connected")
            return False

        try:
            self._instrument.write(":INSTrument:COMBine:OFF")
            time.sleep(0.5)
            self._logger.info("Channel combination mode disabled")
            return True

        except Exception as e:
            self._logger.error(f"Failed to disable combination mode: {e}")
            return False

    def get_combination_mode(self) -> Optional[str]:
        """
        Query the current channel combination mode.

        SCPI Command: :INSTrument:COMBine?

        Returns:
            Combination mode string, or None if query fails
            Possible values: "NONE", "TRACK CH1CH2", "TRACK CH2CH3", 
                           "TRACK CH1CH2CH3", "PARALLEL CH1CH2", 
                           "PARALLEL CH2CH3", "PARALLEL CH1CH2CH3", "SERIES CH1CH2"
        """
        if not self.is_connected:
            self._logger.error("Cannot query combination mode: not connected")
            return None

        try:
            mode = self._instrument.query(":INSTrument:COMBine?").strip()
            self._logger.debug(f"Current combination mode: {mode}")
            return mode

        except Exception as e:
            self._logger.error(f"Failed to query combination mode: {e}")
            return None

    def set_parallel_state(self, channels: str) -> bool:
        """
        Set parallel synchronization state (alternate command).

        SCPI Command: :SOURce:OUTPut:PARallel:STATe {channels}

        Args:
            channels: Channel combination string
                     Valid values: "CH1CH2", "CH2CH3", "CH1CH2CH3"

        Returns:
            True if state set successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set parallel state: not connected")
            return False

        try:
            self._instrument.write(f":SOURce:OUTPut:PARallel:STATe {channels}")
            time.sleep(0.5)
            self._logger.info(f"Parallel state set for {channels}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to set parallel state: {e}")
            return False

    def set_series_state(self, enable: bool) -> bool:
        """
        Enable or disable series synchronization state (alternate command).

        SCPI Command: :SOURce:OUTPut:SERies:STATe {ON|OFF}

        Args:
            enable: True to enable series state, False to disable

        Returns:
            True if state set successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set series state: not connected")
            return False

        try:
            state = "ON" if enable else "OFF"
            self._instrument.write(f":SOURce:OUTPut:SERies:STATe {state}")
            time.sleep(0.5)
            self._logger.info(f"Series state set to {state}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to set series state: {e}")
            return False
    
    # ========================================================================
    # SAVE AND RECALL
    # ========================================================================

    def save_setup(self, location: int) -> bool:
        """
        Save the current instrument configuration to non-volatile memory.

        SCPI Command: *SAV {location}

        Args:
            location: Memory location (1 to 36)

        Returns:
            True if configuration saved successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot save setup: not connected")
            return False

        if not (1 <= location <= 36):
            self._logger.error(f"Invalid memory location {location} (valid range: 1-36)")
            return False

        try:
            self._instrument.write(f"*SAV {location}")
            time.sleep(0.5)
            self._logger.info(f"Configuration saved to memory location {location}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to save setup to location {location}: {e}")
            return False

    def recall_setup(self, location: int) -> bool:
        """
        Recall a previously saved instrument configuration from memory.

        SCPI Command: *RCL {location}

        Args:
            location: Memory location (1 to 36)

        Returns:
            True if configuration recalled successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot recall setup: not connected")
            return False

        if not (1 <= location <= 36):
            self._logger.error(f"Invalid memory location {location} (valid range: 1-36)")
            return False

        try:
            self._instrument.write(f"*RCL {location}")
            time.sleep(1.0)  # Allow time for configuration to be applied
            self._logger.info(f"Configuration recalled from memory location {location}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to recall setup from location {location}: {e}")
            return False

    def reset_to_defaults(self) -> bool:
        """
        Reset the instrument to factory default settings.

        SCPI Command: *RST

        Returns:
            True if reset successful, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot reset instrument: not connected")
            return False

        try:
            self._instrument.write("*RST")
            time.sleep(self._reset_time)
            self._logger.info("Instrument reset to factory defaults")
            return True

        except Exception as e:
            self._logger.error(f"Failed to reset instrument: {e}")
            return False

    # ========================================================================
    # SYSTEM CONTROL COMMANDS
    # ========================================================================

    def set_remote_mode(self) -> bool:
        """
        Switch the instrument to remote control mode (disable front panel).

        SCPI Command: :SYSTem:REMote

        Returns:
            True if mode set successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set remote mode: not connected")
            return False

        try:
            self._instrument.write(":SYSTem:REMote")
            self._logger.info("Instrument switched to remote control mode")
            return True

        except Exception as e:
            self._logger.error(f"Failed to set remote mode: {e}")
            return False

    def set_local_mode(self) -> bool:
        """
        Switch the instrument to local control mode (enable front panel).

        SCPI Command: :SYSTem:LOCal

        Returns:
            True if mode set successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set local mode: not connected")
            return False

        try:
            self._instrument.write(":SYSTem:LOCal")
            self._logger.info("Instrument switched to local control mode")
            return True

        except Exception as e:
            self._logger.error(f"Failed to set local mode: {e}")
            return False

    def lock_remote_mode(self) -> bool:
        """
        Lock the instrument in remote control mode (disable LOCAL button).

        SCPI Command: :SYSTem:RWLock:STATe

        Returns:
            True if lock set successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot lock remote mode: not connected")
            return False

        try:
            self._instrument.write(":SYSTem:RWLock:STATe")
            self._logger.info("Instrument locked in remote control mode")
            return True

        except Exception as e:
            self._logger.error(f"Failed to lock remote mode: {e}")
            return False

    def trigger_beeper(self) -> bool:
        """
        Test the instrument's beeper function.

        SCPI Command: :SYSTem:BEEPer

        Returns:
            True if beeper triggered successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot trigger beeper: not connected")
            return False

        try:
            self._instrument.write(":SYSTem:BEEPer")
            self._logger.debug("Beeper triggered")
            return True

        except Exception as e:
            self._logger.error(f"Failed to trigger beeper: {e}")
            return False

    def self_test(self) -> Optional[int]:
        """
        Execute instrument self-test and return result code.

        SCPI Command: *TST?

        Returns:
            0 if self-test passed, non-zero error code if failed, None on query error
        """
        if not self.is_connected:
            self._logger.error("Cannot run self-test: not connected")
            return None

        try:
            result = int(self._instrument.query("*TST?").strip())

            if result == 0:
                self._logger.info("Self-test passed")
            else:
                self._logger.warning(f"Self-test failed with error code: {result}")

            return result

        except Exception as e:
            self._logger.error(f"Failed to execute self-test: {e}")
            return None

    # ========================================================================
    # ERROR AND STATUS HANDLING
    # ========================================================================

    def get_error(self) -> Optional[Tuple[int, str]]:
        """
        Read and remove the oldest error from the error queue.

        SCPI Command: :SYSTem:ERRor?

        Returns:
            Tuple of (error_code, error_message), or None if query fails
            Error code 0 indicates no error
        """
        if not self.is_connected:
            self._logger.error("Cannot query error: not connected")
            return None

        try:
            response = self._instrument.query(":SYSTem:ERRor?").strip()
            # Response format: "error_code,error_message"
            parts = response.split(',', 1)

            error_code = int(parts[0])
            error_message = parts[1].strip('"') if len(parts) > 1 else "Unknown error"

            if error_code != 0:
                self._logger.warning(f"Error {error_code}: {error_message}")

            return (error_code, error_message)

        except Exception as e:
            self._logger.error(f"Failed to query error: {e}")
            return None

    def clear_status(self) -> bool:
        """
        Clear all status registers and the error queue.

        SCPI Command: *CLS

        Returns:
            True if status cleared successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot clear status: not connected")
            return False

        try:
            self._instrument.write("*CLS")
            time.sleep(0.3)
            self._logger.debug("Status registers and error queue cleared")
            return True

        except Exception as e:
            self._logger.error(f"Failed to clear status: {e}")
            return False

    def wait_for_operation_complete(self) -> bool:
        """
        Wait for all pending operations to complete.

        SCPI Command: *WAI

        Returns:
            True if wait successful, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot wait for operations: not connected")
            return False

        try:
            self._instrument.write("*WAI")
            self._logger.debug("Waiting for all operations to complete")
            return True

        except Exception as e:
            self._logger.error(f"Failed to wait for operation complete: {e}")
            return False

    def query_operation_complete(self) -> Optional[bool]:
        """
        Query whether all pending operations are complete.

        SCPI Command: *OPC?

        Returns:
            True if operations complete, False otherwise, None on query error
        """
        if not self.is_connected:
            self._logger.error("Cannot query operation complete: not connected")
            return None

        try:
            response = self._instrument.query("*OPC?").strip()
            complete = (response == "1")
            self._logger.debug(f"Operations complete: {complete}")
            return complete

        except Exception as e:
            self._logger.error(f"Failed to query operation complete: {e}")
            return None

    def set_operation_complete_on_finish(self) -> bool:
        """
        Set the operation complete bit when all operations finish.

        SCPI Command: *OPC

        Returns:
            True if command sent successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set OPC: not connected")
            return False

        try:
            self._instrument.write("*OPC")
            self._logger.debug("Operation complete bit will be set on finish")
            return True

        except Exception as e:
            self._logger.error(f"Failed to set OPC: {e}")
            return False

    def read_status_byte(self) -> Optional[int]:
        """
        Read the status byte register.

        SCPI Command: *STB?

        Returns:
            Status byte value (0-255), or None if query fails
        """
        if not self.is_connected:
            self._logger.error("Cannot read status byte: not connected")
            return None

        try:
            status = int(self._instrument.query("*STB?").strip())
            self._logger.debug(f"Status byte: {status} (0x{status:02X})")
            return status

        except Exception as e:
            self._logger.error(f"Failed to read status byte: {e}")
            return None

    def read_standard_event_status(self) -> Optional[int]:
        """
        Read and clear the standard event status register.

        SCPI Command: *ESR?

        Returns:
            Standard event status value (0-255), or None if query fails
        """
        if not self.is_connected:
            self._logger.error("Cannot read event status: not connected")
            return None

        try:
            status = int(self._instrument.query("*ESR?").strip())
            self._logger.debug(f"Standard event status: {status} (0x{status:02X})")
            return status

        except Exception as e:
            self._logger.error(f"Failed to read event status: {e}")
            return None

    def set_event_status_enable(self, mask: int) -> bool:
        """
        Set the standard event status enable register mask.

        SCPI Command: *ESE {mask}

        Args:
            mask: Enable mask value (0-255)

        Returns:
            True if mask set successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set event status enable: not connected")
            return False

        if not (0 <= mask <= 255):
            self._logger.error(f"Invalid mask value {mask} (valid range: 0-255)")
            return False

        try:
            self._instrument.write(f"*ESE {mask}")
            self._logger.debug(f"Event status enable mask set to {mask} (0x{mask:02X})")
            return True

        except Exception as e:
            self._logger.error(f"Failed to set event status enable: {e}")
            return False

    def get_event_status_enable(self) -> Optional[int]:
        """
        Query the standard event status enable register mask.

        SCPI Command: *ESE?

        Returns:
            Enable mask value (0-255), or None if query fails
        """
        if not self.is_connected:
            self._logger.error("Cannot query event status enable: not connected")
            return None

        try:
            mask = int(self._instrument.query("*ESE?").strip())
            self._logger.debug(f"Event status enable mask: {mask} (0x{mask:02X})")
            return mask

        except Exception as e:
            self._logger.error(f"Failed to query event status enable: {e}")
            return None

    def set_service_request_enable(self, mask: int) -> bool:
        """
        Set the service request enable register mask.

        SCPI Command: *SRE {mask}

        Args:
            mask: Enable mask value (0-255)

        Returns:
            True if mask set successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set service request enable: not connected")
            return False

        if not (0 <= mask <= 255):
            self._logger.error(f"Invalid mask value {mask} (valid range: 0-255)")
            return False

        try:
            self._instrument.write(f"*SRE {mask}")
            self._logger.debug(f"Service request enable mask set to {mask} (0x{mask:02X})")
            return True

        except Exception as e:
            self._logger.error(f"Failed to set service request enable: {e}")
            return False

    def get_service_request_enable(self) -> Optional[int]:
        """
        Query the service request enable register mask.

        SCPI Command: *SRE?

        Returns:
            Enable mask value (0-255), or None if query fails
        """
        if not self.is_connected:
            self._logger.error("Cannot query service request enable: not connected")
            return None

        try:
            mask = int(self._instrument.query("*SRE?").strip())
            self._logger.debug(f"Service request enable mask: {mask} (0x{mask:02X})")
            return mask

        except Exception as e:
            self._logger.error(f"Failed to query service request enable: {e}")
            return None

    def set_power_on_status_clear(self, clear_on_power_on: bool) -> bool:
        """
        Configure whether status registers are cleared on power-on.

        SCPI Command: *PSC {0|1}

        Args:
            clear_on_power_on: True to clear registers on power-on, False to preserve

        Returns:
            True if setting applied successfully, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot set power-on status clear: not connected")
            return False

        try:
            value = 1 if clear_on_power_on else 0
            self._instrument.write(f"*PSC {value}")
            self._logger.debug(f"Power-on status clear set to {clear_on_power_on}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to set power-on status clear: {e}")
            return False

    def get_power_on_status_clear(self) -> Optional[bool]:
        """
        Query whether status registers are cleared on power-on.

        SCPI Command: *PSC?

        Returns:
            True if cleared on power-on, False if preserved, None on query error
        """
        if not self.is_connected:
            self._logger.error("Cannot query power-on status clear: not connected")
            return None

        try:
            response = self._instrument.query("*PSC?").strip()
            clear_on_power_on = (response == "1")
            self._logger.debug(f"Power-on status clear: {clear_on_power_on}")
            return clear_on_power_on

        except Exception as e:
            self._logger.error(f"Failed to query power-on status clear: {e}")
            return None

    def read_operation_condition(self) -> Optional[int]:
        """
        Read the operation condition register (real-time status).

        SCPI Command: :STATus:OPERation:CONDition?

        Returns:
            Operation condition register value (0-255), or None if query fails
        """
        if not self.is_connected:
            self._logger.error("Cannot read operation condition: not connected")
            return None

        try:
            condition = int(self._instrument.query(":STATus:OPERation:CONDition?").strip())
            self._logger.debug(f"Operation condition: {condition} (0x{condition:02X})")
            return condition

        except Exception as e:
            self._logger.error(f"Failed to read operation condition: {e}")
            return None

    def read_operation_event(self) -> Optional[int]:
        """
        Read and clear the operation event register (latched status).

        SCPI Command: :STATus:OPERation:EVENt?

        Returns:
            Operation event register value (0-255), or None if query fails
        """
        if not self.is_connected:
            self._logger.error("Cannot read operation event: not connected")
            return None

        try:
            event = int(self._instrument.query(":STATus:OPERation:EVENt?").strip())
            self._logger.debug(f"Operation event: {event} (0x{event:02X})")
            return event

        except Exception as e:
            self._logger.error(f"Failed to read operation event: {e}")
            return None

    def read_questionable_condition(self) -> Optional[int]:
        """
        Read the questionable condition register (real-time status).

        SCPI Command: :STATus:QUEStionable:CONDition?

        Returns:
            Questionable condition register value (0-255), or None if query fails
        """
        if not self.is_connected:
            self._logger.error("Cannot read questionable condition: not connected")
            return None

        try:
            condition = int(self._instrument.query(":STATus:QUEStionable:CONDition?").strip())
            self._logger.debug(f"Questionable condition: {condition} (0x{condition:02X})")
            return condition

        except Exception as e:
            self._logger.error(f"Failed to read questionable condition: {e}")
            return None

    def read_questionable_event(self) -> Optional[int]:
        """
        Read and clear the questionable event register (latched status).

        SCPI Command: :STATus:QUEStionable:EVENt?

        Returns:
            Questionable event register value (0-255), or None if query fails
        """
        if not self.is_connected:
            self._logger.error("Cannot read questionable event: not connected")
            return None

        try:
            event = int(self._instrument.query(":STATus:QUEStionable:EVENt?").strip())
            self._logger.debug(f"Questionable event: {event} (0x{event:02X})")
            return event

        except Exception as e:
            self._logger.error(f"Failed to read questionable event: {e}")
            return None

    def preset_status_registers(self) -> bool:
        """
        Reset all status register bits to default state.

        SCPI Command: :STATus:PRESet

        Returns:
            True if reset successful, False otherwise
        """
        if not self.is_connected:
            self._logger.error("Cannot preset status registers: not connected")
            return False

        try:
            self._instrument.write(":STATus:PRESet")
            time.sleep(0.3)
            self._logger.debug("Status registers preset to defaults")
            return True

        except Exception as e:
            self._logger.error(f"Failed to preset status registers: {e}")
            return False


# ============================================================================
# USAGE EXAMPLES AND MODULE DOCUMENTATION
# ============================================================================

def example_basic_usage():
    """
    Example demonstrating basic power supply control operations.
    """
    # Initialize and connect
    psu = KeithleyPowerSupply("USB0::0x05E6::0x2230::INSTR")

    if not psu.connect():
        print("Failed to connect to power supply")
        return

    try:
        # Get instrument information
        info = psu.get_instrument_info()
        if info is None:
            print("Warning: Could not retrieve instrument information")
            print("Continuing with default settings...")
        else:
            print(f"Connected to {info.get('manufacturer', 'Unknown')} {info.get('model', 'power supply')}")

        # Configure channel 1: 5V, 1A limit, 6V OVP
        psu.configure_channel(1, voltage=5.0, current_limit=1.0, ovp_level=6.0)

        # Enable output
        psu.enable_channel_output(1)

        # Measure output
        try:
            measurement = psu.measure_channel_output(1)
            if measurement is not None:
                voltage, current = measurement
                print(f"CH1: {voltage:.3f}V, {current:.3f}A")
            else:
                print("Failed to measure output: No data received")
        except Exception as e:
            print(f"Failed to measure output: {e}")

        # Disable output
        psu.disable_channel_output(1)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Always disconnect when done
        psu.disconnect()


def example_advanced_usage():
    """
    Example demonstrating advanced features including combinations and timers.
    """
    psu = KeithleyPowerSupply("USB0::0x05E6::0x2230::INSTR")

    if not psu.connect():
        return

    try:
        # Configure multiple channels
        psu.apply_channel_settings(1, voltage=12.0, current_limit=2.0)
        psu.apply_channel_settings(2, voltage=12.0, current_limit=2.0)

        # Set parallel mode to combine current capacity
        psu.set_parallel_mode("CH1CH2")

        # Enable delayed output (5 second delay)
        psu.set_output_timer_delay(1, 5.0)
        psu.enable_output_timer(1, True)

        # Enable output (will turn on after 5 seconds)
        psu.enable_channel_output(1)

        # Save configuration to memory
        psu.save_setup(1)

        # Measure all channels
        measurements = psu.measure_all_channels()
        if measurements is not None:
            for ch, data in measurements.items():
                print(f"{ch}: {data['voltage']:.3f}V, {data['current']:.3f}A, {data['power']:.3f}W")
        else:
            print("Warning: Failed to measure all channels")

    finally:
        psu.disable_all_outputs()
        psu.disconnect()


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run example
    example_basic_usage()
