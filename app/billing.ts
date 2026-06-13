// billing.ts - React Native Billing Forecasting and Tariff Calculations
// Smart Home Energy Optimization System

export interface UsageDataPoint {
  day: number;            // Day of month (1 to 30)
  consumptionKWh: number; // Cumulative kWh consumed up to this day
}

export interface SlabConfig {
  limit: number;          // Cumulative unit limit (kWh)
  rate: number;           // Cost per unit (PKR)
}

// Official Pakistani Domestic Residential Tariff Slabs
export const DOMESTIC_SLABS_PKR: SlabConfig[] = [
  { limit: 100, rate: 22.0 },    // Slab 1: First 100 units
  { limit: 200, rate: 28.5 },    // Slab 2: 101 to 200 units
  { limit: 300, rate: 34.0 },    // Slab 3: 201 to 300 units
  { limit: 700, rate: 42.0 },    // Slab 4: 301 to 700 units
  { limit: Infinity, rate: 48.0 } // Slab 5: Above 700 units (Peak residential rate)
];

// National Taxes and Fees
export const TAX_GST_RATE = 0.17;          // 17% General Sales Tax
export const TAX_DUTY_RATE = 0.015;        // 1.5% State Electricity Duty
export const METER_FIXED_CHARGES = 350;    // Fixed monthly service charges in PKR

/**
 * Predicts month-end energy consumption (kWh) using Ordinary Least Squares (OLS) Linear Regression.
 * Fits: y = mx + c (where x is Day, y is Cumulative kWh). Projects y at x = 30.
 * 
 * @param history Array of daily cumulative consumption data points
 * @param currentDay The active day of the billing cycle (1 to 30)
 * @returns Predicted month-end cumulative units (kWh)
 */
export function forecastMonthlyUsage(history: UsageDataPoint[], currentDay: number): number {
  if (history.length < 2) {
    // Fallback: Simple linear extrapolation if history is insufficient
    const currentUsage = history[history.length - 1]?.consumptionKWh || 0;
    return currentDay > 0 ? (currentUsage / currentDay) * 30 : 0;
  }

  const n = history.length;
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumXX = 0;

  for (const point of history) {
    sumX += point.day;
    sumY += point.consumptionKWh;
    sumXY += point.day * point.consumptionKWh;
    sumXX += point.day * point.day;
  }

  // Calculate Slope (m) and Intercept (c)
  const numeratorSlope = n * sumXY - sumX * sumY;
  const denominatorSlope = n * sumXX - sumX * sumX;
  
  // Guard against division by zero (e.g. all X coordinates are identical)
  const slope = denominatorSlope !== 0 ? numeratorSlope / denominatorSlope : 0;
  const intercept = (sumY - slope * sumX) / n;

  // Project value at Day 30
  const prediction = slope * 30 + intercept;

  // Safety floor check: Forecast cannot be lower than what has already been consumed
  const currentUsage = history[history.length - 1].consumptionKWh;
  return Math.max(prediction, currentUsage);
}

/**
 * Calculates the total electricity bill in Pakistani Rupees (PKR) based on slab tariff and national taxes.
 * 
 * @param forecastedUnits The forecasted monthly units (kWh)
 * @returns Total estimated bill amount in PKR (rounded to nearest integer)
 */
export function calculateWAPDATariffBill(forecastedUnits: number): number {
  if (forecastedUnits <= 0) return METER_FIXED_CHARGES;

  let remainingUnits = forecastedUnits;
  let energyCost = 0;
  let previousLimit = 0;

  // Compute tiered slab costs
  for (const slab of DOMESTIC_SLABS_PKR) {
    const slabCapacity = slab.limit - previousLimit;
    if (remainingUnits > slabCapacity) {
      energyCost += slabCapacity * slab.rate;
      remainingUnits -= slabCapacity;
      previousLimit = slab.limit;
    } else {
      energyCost += remainingUnits * slab.rate;
      break;
    }
  }

  // Calculate taxes and duties on the basic energy cost
  const taxesAmount = energyCost * (TAX_GST_RATE + TAX_DUTY_RATE);
  const totalBillPKR = energyCost + taxesAmount + METER_FIXED_CHARGES;

  return Math.round(totalBillPKR);
}
