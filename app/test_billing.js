// test_billing.js - Test suite for forecasting and WAPDA Slab Billing
const fs = require('fs');
const path = require('path');

// Read the TypeScript file and convert imports/exports to CommonJS to run directly in Node.js
const tsFilePath = path.join(__dirname, 'billing.ts');
let tsContent = fs.readFileSync(tsFilePath, 'utf8');

// Strip TypeScript annotations and export statements for direct Node execution
let jsContent = tsContent
  .replace(/export interface/g, 'interface')
  .replace(/export const/g, 'const')
  .replace(/export function/g, 'function')
  .replace(/: UsageDataPoint\[\]/g, '')
  .replace(/: number/g, '')
  .replace(/: SlabConfig\[\]/g, '')
  .replace(/: number/g, '')
  .replace(/interface UsageDataPoint \{[^}]*\}/g, '')
  .replace(/interface SlabConfig \{[^}]*\}/g, '');

// Append CommonJS exports to the stripped code
jsContent += `
module.exports = {
  forecastMonthlyUsage,
  calculateWAPDATariffBill,
  DOMESTIC_SLABS_PKR,
  TAX_GST_RATE,
  TAX_DUTY_RATE,
  METER_FIXED_CHARGES
};
`;

// Evaluate the code dynamically in a sandboxed style
const moduleFn = new Function('module', 'exports', 'require', '__dirname', '__filename', jsContent);
const mockModule = { exports: {} };
moduleFn(mockModule, mockModule.exports, require, __dirname, tsFilePath);

const { forecastMonthlyUsage, calculateWAPDATariffBill } = mockModule.exports;

// --- Test Suite Execution ---
console.log("=========================================");
console.log("RUNNING BILLING ENGINE MATH VERIFICATION");
console.log("=========================================\n");

let passedTests = 0;
let failedTests = 0;

function assertEqual(actual, expected, message) {
  if (Math.abs(actual - expected) < 1e-4) {
    console.log(`[PASS] ${message} (Result: ${actual})`);
    passedTests++;
  } else {
    console.error(`[FAIL] ${message} (Expected: ${expected}, Got: ${actual})`);
    failedTests++;
  }
}

try {
  // Test Case 1: Constant consumption forecasting
  // Daily consumption of exactly 10 kWh. Day 10 cumulative is 100 kWh.
  const history1 = [
    { day: 1, consumptionKWh: 10 },
    { day: 2, consumptionKWh: 20 },
    { day: 3, consumptionKWh: 30 },
    { day: 4, consumptionKWh: 40 },
    { day: 5, consumptionKWh: 50 },
    { day: 10, consumptionKWh: 100 }
  ];
  const prediction1 = forecastMonthlyUsage(history1, 10);
  assertEqual(prediction1, 300, "Linear forecasting for constant 10 kWh daily usage");

  // Test Case 2: Escalating consumption forecasting
  // Day 1 to 5 cumulative kWh: 10, 25, 45, 70, 100 (increasing daily increments: 10, 15, 20, 25, 30)
  const history2 = [
    { day: 1, consumptionKWh: 10 },
    { day: 2, consumptionKWh: 25 },
    { day: 3, consumptionKWh: 45 },
    { day: 4, consumptionKWh: 70 },
    { day: 5, consumptionKWh: 100 }
  ];
  const prediction2 = forecastMonthlyUsage(history2, 5);
  // OLS Linear Regression fit for y values: [10, 25, 45, 70, 100] at x: [1, 2, 3, 4, 5]
  // Slope m = 22.5, Intercept c = -17.5. At Day 30: y = 22.5 * 30 - 17.5 = 675 - 17.5 = 657.5
  assertEqual(prediction2, 657.5, "OLS regression projection at Day 30 for increasing usage");

  // Test Case 3: Slab billing logic below first threshold
  // Forecasted units = 50 kWh. Falls completely under Slab 1 (50 units * 22.0 PKR/unit = 1100 PKR)
  // Taxes = 1100 * (0.17 + 0.015) = 1100 * 0.185 = 203.5 PKR
  // Total = 1100 + 203.5 + 350 (fixed) = 1653.5 PKR (rounded to 1654)
  const bill1 = calculateWAPDATariffBill(50);
  assertEqual(bill1, 1654, "Slab billing tariff calculation for 50 kWh (below threshold)");

  // Test Case 4: Slab billing crossing multiple thresholds
  // Forecasted units = 250 kWh.
  // Slab 1 (first 100 units): 100 * 22.0 = 2200 PKR
  // Slab 2 (next 100 units): 100 * 28.5 = 2850 PKR
  // Slab 3 (remaining 50 units): 50 * 34.0 = 1700 PKR
  // Total basic cost = 2200 + 2850 + 1700 = 6750 PKR
  // Taxes = 6750 * 0.185 = 1248.75 PKR
  // Total bill = 6750 + 1248.75 + 350 (fixed) = 8348.75 PKR (rounded to 8349)
  const bill2 = calculateWAPDATariffBill(250);
  assertEqual(bill2, 8349, "Tiered slab tariff calculation for 250 kWh crossing three slabs");

  // Test Case 5: Zero consumption
  // Forecasted units = 0. Should return only fixed charges.
  const bill3 = calculateWAPDATariffBill(0);
  assertEqual(bill3, 350, "Slab billing for 0 kWh (returns fixed meter fee)");

} catch (error) {
  console.error("Critical test runner failure:", error);
  failedTests++;
}

console.log("\n-----------------------------------------");
console.log(`TEST SUMMARY: Passed: ${passedTests}, Failed: ${failedTests}`);
console.log("-----------------------------------------\n");

if (failedTests > 0) {
  process.exit(1);
} else {
  process.exit(0);
}
