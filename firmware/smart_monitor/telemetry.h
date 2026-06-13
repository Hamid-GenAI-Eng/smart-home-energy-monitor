// telemetry.h - Packed structs for telemetry logging and data broadcasting
// Smart Home Energy Optimization System

#ifndef TELEMETRY_H
#define TELEMETRY_H

#include <Arduino.h>

// Strictly packed to exactly 40 bytes to align with SD Card physical block boundaries (512 bytes)
// 512 bytes / 40 bytes = 12 structs per block with only 32 bytes of padding at block end, maximizing write endurance.
struct __attribute__((__packed__)) EnergyTelemetry {
    uint32_t timestamp;      // Epoch Unix time (4 bytes)
    float voltage;           // Line Voltage in Volts (4 bytes)
    float current;           // Current in Amperes (4 bytes)
    float active_power;      // Active Power in Watts (4 bytes)
    float frequency;         // Line Frequency in Hz (4 bytes)
    float power_factor;      // Power Factor 0.0 to 1.0 (4 bytes)
    float energy_kwh;        // Cumulative Energy consumption (4 bytes)
    float temperature;       // Local ambient temperature (4 bytes)
    float humidity;          // Local ambient humidity (4 bytes)
    uint16_t ups_charge;     // UPS backup battery charge percentage (2 bytes)
    uint16_t status_flags;   // Bitwise system state (Relay status, Leakage, AI alert) (2 bytes)
};

// Bitmasks for system status flags
#define FLAG_RELAY_ON        (1 << 0)   // Bit 0: Main relay status (0 = Cut/Open, 1 = Connected/Closed)
#define FLAG_POWER_LEAKAGE   (1 << 1)   // Bit 1: Background leakage detected (>10W draw when no active load is running)
#define FLAG_FAULT_OVERLOAD  (1 << 2)   // Bit 2: Overload trip triggered (current exceeds threshold, e.g. 15A)
#define FLAG_FAULT_BROWNOUT  (1 << 3)   // Bit 3: Brownout trip triggered (voltage drops below threshold, e.g. 180V)
#define FLAG_FAULT_OVERVOLT  (1 << 4)   // Bit 4: Overvoltage trip triggered (voltage exceeds threshold, e.g. 250V)
#define FLAG_AI_FRIDGE_ON    (1 << 5)   // Bit 5: Refrigerator ON state (TinyML disaggregation)
#define FLAG_AI_MOTOR_ON     (1 << 6)   // Bit 6: Water Motor ON state (TinyML disaggregation)
#define FLAG_AI_IRON_ON      (1 << 7)   // Bit 7: Iron / Heavy AC load ON state (TinyML disaggregation)

#endif // TELEMETRY_H
