"""
LCD Hardware Calibration Tool

Purpose: Measure real-world LCD parameters for optical_classifier_amplitude_REALISTIC.py

What This Tool Does:
1. Displays test patterns on your LCD screen
2. Guides you through measurements using simple equipment
3. Generates lcd_calibration.json with measured values

Equipment Needed:
- Your 1.54" LCD screen (200×200, monochrome reflective)
- Photodetector or light meter (or smartphone lux meter app)
- USB microscope or magnifying glass (optional, for fill factor)
- Green laser pointer (532nm, optional for better measurements)

Measurements to Perform:
1. Fill Factor: Measure pixel gap sizes with microscope
2. Contrast Ratio: Measure min/max light transmission with photodetector
3. Gap Transmission: Measure uncontrolled light leakage through gaps
4. Response Time: Measure switching speed (optional, not used in static mask)

Output: lcd_calibration.json (used by optical_classifier_amplitude_REALISTIC.py)

Philosophy: Start with ideal assumptions → measure real hardware → update parameters
"""

import numpy as np
import json
import time
from pathlib import Path

# ============================================================
# SINGLE LCD SPECIFICATIONS (Known from Datasheet)
# ============================================================

LCD_SPECS = {
    'resolution': (200, 200),
    'active_area': (27.66e-3, 27.66e-3),  # 27.66mm
    'dot_pitch': 138.3e-6,  # 138.3 μm
}

# ============================================================
# CALIBRATION PARAMETERS (UNKNOWN - Measured with this tool)
# ============================================================
# These start as IDEAL assumptions, then updated with real measurements:
#
# 1. fill_factor: Ratio of active pixel area (start: 1.0 = 100% IDEAL)
# 2. gap_transmission: Light leakage through gaps (start: 0.5 = conservative)
# 3. contrast_ratio: Max/min transmission ratio (start: inf = IDEAL)
# 4. min_transmission: Black pixel transmission (start: 0.0 = IDEAL)
# 5. max_transmission: White pixel transmission (start: 1.0 = IDEAL)
#
# Philosophy: IDEAL → MEASURE → UPDATE → TRAIN WITH REALITY
# ============================================================

print("="*70)
print("LCD HARDWARE CALIBRATION TOOL")
print("="*70)
print("\nThis tool will guide you through measuring your LCD's real parameters.")
print(f"\nLCD Specs:")
print(f"  Resolution: {LCD_SPECS['resolution'][0]}×{LCD_SPECS['resolution'][1]} pixels")
print(f"  Pixel pitch: {LCD_SPECS['dot_pitch']*1e6:.1f} μm")
print(f"  Active area: {LCD_SPECS['active_area'][0]*1e3:.2f}mm × {LCD_SPECS['active_area'][1]*1e3:.2f}mm")

print(f"\nPhilosophy:")
print(f"  • Start: Ideal assumptions (100% fill, infinite contrast)")
print(f"  • Measure: Real hardware with simple tools")
print(f"  • Update: Parameters to match reality")

# ============================================================
# CALIBRATION 1: FILL FACTOR (Pixel Gap Measurement)
# ============================================================

def calibrate_fill_factor():
    """
    Measure the fill factor (ratio of active pixel area to total area).
    
    Methods:
    1. MICROSCOPE METHOD (Most Accurate):
       - Use USB microscope to photograph pixels
       - Measure: pixel_width and gap_width
       - Calculate: fill_factor = (pixel_width / (pixel_width + gap_width))²
    
    2. VISUAL ESTIMATION METHOD:
       - Typical LCD: 70-85% fill factor
       - High-quality: 85-95%
       - Low-quality: 60-75%
    
    3. SKIP (Use ideal assumption): fill_factor = 1.0
    """
    print("\n" + "="*70)
    print("CALIBRATION 1: FILL FACTOR (Pixel Gap Measurement)")
    print("="*70)
    
    print("\nWhat is Fill Factor?")
    print("  LCD pixels have gaps between them. Fill factor = active_area / total_area")
    print("  Example: If 80% is active pixel, 20% is gaps → fill_factor = 0.80")
    
    print("\nMeasurement Methods:")
    print("  [1] MICROSCOPE: Measure pixel and gap widths (RECOMMENDED)")
    print("  [2] ESTIMATE: Typical values based on LCD quality")
    print("  [3] SKIP: Assume ideal (100% fill factor, no gaps)")
    
    while True:
        choice = input("\nSelect method [1/2/3]: ").strip()
        
        if choice == '1':
            print("\n📸 MICROSCOPE METHOD:")
            print("\nSteps:")
            print("  1. Display a white pattern on your LCD")
            print("  2. Use USB microscope to photograph pixels")
            print("  3. Measure in the photo:")
            print("     - pixel_width: Width of one pixel (bright area)")
            print("     - gap_width: Width of gap between pixels (dark area)")
            print("  4. Calculate: fill_factor = (pixel_width / (pixel_width + gap_width))²")
            
            print("\nExample:")
            print("  If pixel_width = 120μm, gap_width = 18.3μm")
            print(f"  Then total pitch = 120 + 18.3 = 138.3μm ✓ (matches your LCD!)")
            print(f"  Fill factor = (120/138.3)² = 0.752 = 75.2%")
            
            pixel_width = float(input("\nEnter measured pixel_width (μm): ")) * 1e-6
            gap_width = float(input("Enter measured gap_width (μm): ")) * 1e-6
            
            measured_pitch = pixel_width + gap_width
            expected_pitch = LCD_SPECS['dot_pitch']
            
            print(f"\n✓ Measured pitch: {measured_pitch*1e6:.2f} μm")
            print(f"  Expected pitch: {expected_pitch*1e6:.2f} μm")
            
            if abs(measured_pitch - expected_pitch) / expected_pitch > 0.05:
                print(f"⚠ WARNING: Measured pitch differs from spec by {abs(measured_pitch - expected_pitch)/expected_pitch*100:.1f}%")
                print(f"  Please remeasure or check your measurements.")
                continue
            
            fill_factor_1d = pixel_width / measured_pitch
            fill_factor = fill_factor_1d ** 2
            
            print(f"\n✓ Fill factor (1D): {fill_factor_1d:.3f}")
            print(f"✓ Fill factor (2D): {fill_factor:.3f}")
            
            return fill_factor
        
        elif choice == '2':
            print("\n📊 ESTIMATE METHOD:")
            print("\nTypical fill factors by LCD quality:")
            print("  High-quality (expensive): 0.85 - 0.95")
            print("  Medium-quality (typical): 0.75 - 0.85")
            print("  Low-quality (cheap): 0.60 - 0.75")
            
            print("\nYour LCD: 1.54\" monochrome reflective")
            print("  Likely range: 0.70 - 0.85 (medium quality)")
            
            while True:
                fill_factor = float(input("\nEnter estimated fill_factor [0.6-0.95]: "))
                if 0.5 <= fill_factor <= 1.0:
                    return fill_factor
                print("⚠ Invalid range. Must be between 0.5 and 1.0")
        
        elif choice == '3':
            print("\n⚠ SKIPPING: Assuming ideal (fill_factor = 1.0)")
            return 1.0
        
        else:
            print("⚠ Invalid choice. Please enter 1, 2, or 3.")


# ============================================================
# CALIBRATION 2: CONTRAST RATIO (Transmission Measurement)
# ============================================================

def calibrate_contrast_ratio():
    """
    Measure the contrast ratio (ratio of max transmission to min transmission).
    
    Methods:
    1. PHOTODETECTOR METHOD (Most Accurate):
       - Use photodetector or light meter
       - Display white pattern → measure I_max
       - Display black pattern → measure I_min
       - Calculate: contrast_ratio = I_max / I_min
    
    2. VISUAL ESTIMATION METHOD:
       - Typical monochrome reflective LCD: 5:1 to 20:1
       - High-quality: 15:1 to 30:1
    
    3. SKIP (Use ideal assumption): contrast_ratio = inf
    """
    print("\n" + "="*70)
    print("CALIBRATION 2: CONTRAST RATIO (Transmission Measurement)")
    print("="*70)
    
    print("\nWhat is Contrast Ratio?")
    print("  How much more light passes through 'white' vs 'black' pixels")
    print("  Contrast ratio = max_transmission / min_transmission")
    print("  Example: 10:1 means white is 10× brighter than black")
    
    print("\nMeasurement Methods:")
    print("  [1] PHOTODETECTOR: Measure light intensity (RECOMMENDED)")
    print("  [2] ESTIMATE: Typical values for monochrome reflective LCDs")
    print("  [3] SKIP: Assume ideal (infinite contrast)")
    
    while True:
        choice = input("\nSelect method [1/2/3]: ").strip()
        
        if choice == '1':
            print("\n📊 PHOTODETECTOR METHOD:")
            print("\nEquipment:")
            print("  - Photodetector, light meter, or smartphone lux meter app")
            print("  - Green light source (532nm laser pointer preferred)")
            
            print("\nSetup:")
            print("  1. Place light source on one side of LCD")
            print("  2. Place detector on opposite side")
            print("  3. Align so light passes through LCD center")
            
            print("\nMeasurement Steps:")
            print("  Step A: Display ALL WHITE pattern on LCD")
            print("          → Measure intensity I_max (lux or arbitrary units)")
            print("  Step B: Display ALL BLACK pattern on LCD")
            print("          → Measure intensity I_min (lux or arbitrary units)")
            print("  Step C: Calculate contrast = I_max / I_min")
            
            input("\n[Press Enter when ready to measure I_max (white pattern)]")
            print("📱 Display WHITE pattern on your LCD now...")
            time.sleep(2)
            I_max = float(input("Enter measured I_max (any units): "))
            
            input("\n[Press Enter when ready to measure I_min (black pattern)]")
            print("📱 Display BLACK pattern on your LCD now...")
            time.sleep(2)
            I_min = float(input("Enter measured I_min (same units): "))
            
            if I_min >= I_max:
                print("⚠ ERROR: I_min should be less than I_max!")
                print("  Please check your measurements.")
                continue
            
            contrast_ratio = I_max / I_min
            
            print(f"\n✓ Measured contrast ratio: {contrast_ratio:.1f}:1")
            
            # Calculate actual transmission values
            # Normalize so max_transmission is close to 1.0
            max_transmission = 1.0
            min_transmission = max_transmission / contrast_ratio
            
            print(f"✓ Max transmission: {max_transmission:.3f}")
            print(f"✓ Min transmission: {min_transmission:.3f}")
            
            return contrast_ratio, min_transmission, max_transmission
        
        elif choice == '2':
            print("\n📊 ESTIMATE METHOD:")
            print("\nTypical contrast ratios for monochrome reflective LCDs:")
            print("  Low-quality: 5:1 to 10:1")
            print("  Medium-quality: 10:1 to 20:1")
            print("  High-quality: 20:1 to 50:1")
            
            print("\nYour LCD: Monochrome reflective")
            print("  Likely range: 5:1 to 15:1 (conservative estimate)")
            
            while True:
                contrast_ratio = float(input("\nEnter estimated contrast ratio (e.g., 10 for 10:1): "))
                if contrast_ratio >= 1.0:
                    max_transmission = 1.0
                    min_transmission = max_transmission / contrast_ratio
                    
                    print(f"\n✓ Contrast ratio: {contrast_ratio:.1f}:1")
                    print(f"✓ Max transmission: {max_transmission:.3f}")
                    print(f"✓ Min transmission: {min_transmission:.3f}")
                    
                    return contrast_ratio, min_transmission, max_transmission
                print("⚠ Invalid. Contrast ratio must be ≥ 1.0")
        
        elif choice == '3':
            print("\n⚠ SKIPPING: Assuming ideal (infinite contrast)")
            return float('inf'), 0.0, 1.0
        
        else:
            print("⚠ Invalid choice. Please enter 1, 2, or 3.")


# ============================================================
# CALIBRATION 3: GAP TRANSMISSION (Gap Leakage Measurement)
# ============================================================

def calibrate_gap_transmission(fill_factor, min_transmission, max_transmission):
    """
    Measure the transmission of uncontrolled gaps between pixels.
    
    This is NEW and IMPORTANT! Gaps don't respond to voltage - they leak light.
    
    Methods:
    1. INDIRECT CALCULATION (From previous measurements):
       - Use fill_factor and contrast measurements
       - Estimate gap_transmission = 0.5 (typical)
    
    2. DIRECT MEASUREMENT (Advanced, requires microscope + photodetector):
       - Shine light specifically on gap region
       - Measure transmission directly
    
    3. SKIP (Use conservative default): gap_transmission = 0.5
    """
    print("\n" + "="*70)
    print("CALIBRATION 3: GAP TRANSMISSION (Gap Leakage Measurement)")
    print("="*70)
    
    print("\nWhat is Gap Transmission?")
    print("  Gaps are UNCONTROLLED - they don't respond to LCD voltage")
    print("  They leak light at some constant level, independent of pixel state")
    print("  gap_transmission = how much light passes through gap area")
    
    print("\nWhy This Matters:")
    print("  • Pixels you control: can be BLACK or WHITE")
    print("  • Gaps you DON'T control: always transmit at gap_transmission")
    print("  • This creates a 'floor' - you can't achieve perfect black!")
    
    print("\nMeasurement Methods:")
    print("  [1] ESTIMATE: Use typical value (0.4-0.7 for reflective LCD)")
    print("  [2] CALCULATE: Derive from contrast + fill factor")
    print("  [3] SKIP: Use conservative default (0.5)")
    
    while True:
        choice = input("\nSelect method [1/2/3]: ").strip()
        
        if choice == '1':
            print("\n📊 ESTIMATE METHOD:")
            print("\nTypical gap transmission by LCD type:")
            print("  Transmissive (backlit): 0.3 - 0.6 (glass substrate)")
            print("  Reflective (your LCD): 0.4 - 0.7 (metallic reflectors)")
            print("  High-quality: 0.2 - 0.4 (absorptive coatings)")
            
            print("\nYour LCD: Monochrome reflective")
            print("  Likely range: 0.4 - 0.6 (conservative: 0.5)")
            
            while True:
                gap_transmission = float(input("\nEnter estimated gap_transmission [0.2-0.8]: "))
                if 0.1 <= gap_transmission <= 0.9:
                    print(f"\n✓ Gap transmission: {gap_transmission:.3f}")
                    return gap_transmission
                print("⚠ Invalid range. Must be between 0.1 and 0.9")
        
        elif choice == '2':
            print("\n🧮 CALCULATE METHOD:")
            print("\nUsing your measurements:")
            print(f"  Fill factor: {fill_factor:.3f}")
            print(f"  Min transmission (black): {min_transmission:.3f}")
            print(f"  Max transmission (white): {max_transmission:.3f}")
            
            if fill_factor >= 0.99:
                print("\n⚠ Fill factor is too close to 1.0 - cannot calculate gaps!")
                print("  Use ESTIMATE method instead.")
                continue
            
            # Estimate: assume gaps contribute significantly to min_transmission
            # min_transmission ≈ black_pixel × fill + gap_trans × (1-fill)
            # If black_pixel ≈ 0, then: gap_trans ≈ min_trans / (1-fill)
            estimated_gap_trans = min_transmission / (1 - fill_factor)
            
            print(f"\n🔍 Estimated gap transmission: {estimated_gap_trans:.3f}")
            print(f"  (Assumes black pixels block all light, gaps leak)")
            
            if estimated_gap_trans > 1.0:
                print("\n⚠ WARNING: Estimated value > 1.0 (impossible!)")
                print("  This suggests measurement error or incorrect assumptions.")
                print("  Recommend using ESTIMATE method with 0.5")
                continue
            
            confirm = input(f"\nUse calculated value {estimated_gap_trans:.3f}? [y/n]: ")
            if confirm.lower() == 'y':
                return estimated_gap_trans
        
        elif choice == '3':
            print("\n⚠ SKIPPING: Using conservative default (gap_transmission = 0.5)")
            return 0.5
        
        else:
            print("⚠ Invalid choice. Please enter 1, 2, or 3.")


# ============================================================
# SAVE CALIBRATION
# ============================================================

def save_calibration(fill_factor, contrast_ratio, min_transmission, max_transmission,
                     gap_transmission, output_file='lcd_calibration.json'):
    """
    Save calibration data to JSON file.
    """
    calibration = {
        'fill_factor': float(fill_factor),
        'gap_transmission': float(gap_transmission),
        'contrast_ratio': float(contrast_ratio) if contrast_ratio != float('inf') else None,
        'min_transmission': float(min_transmission),
        'max_transmission': float(max_transmission),
        'calibrated': True,
        'lcd_specs': {
            'resolution': LCD_SPECS['resolution'],
            'dot_pitch_um': LCD_SPECS['dot_pitch'] * 1e6,
            'active_area_mm': [LCD_SPECS['active_area'][0] * 1e3, LCD_SPECS['active_area'][1] * 1e3]
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(calibration, f, indent=2)
    
    print(f"\n✓ Calibration saved to {output_file}")
    return calibration


# ============================================================
# MAIN CALIBRATION WORKFLOW
# ============================================================

def main():
    """
    Main calibration workflow.
    """
    
    # Calibration 1: Fill Factor
    fill_factor = calibrate_fill_factor()
    
    # Calibration 2: Contrast Ratio
    contrast_ratio, min_transmission, max_transmission = calibrate_contrast_ratio()
    
    # Calibration 3: Gap Transmission (NEW!)
    gap_transmission = calibrate_gap_transmission(fill_factor, min_transmission, max_transmission)
    
    # Summary
    print("\n" + "="*70)
    print("CALIBRATION SUMMARY")
    print("="*70)
    print(f"\nMeasured Parameters:")
    print(f"  Fill factor: {fill_factor:.3f} ({fill_factor*100:.1f}% active area)")
    print(f"  Gap transmission: {gap_transmission:.3f} (constant leakage)")
    if contrast_ratio == float('inf'):
        print(f"  Contrast ratio: Infinite (ideal)")
    else:
        print(f"  Contrast ratio: {contrast_ratio:.1f}:1")
    print(f"  Transmission range: [{min_transmission:.3f}, {max_transmission:.3f}]")
    
    # Calculate effective transmission with fill factor + gaps
    print(f"\nEffective Transmission (with hardware effects):")
    effective_black = min_transmission * fill_factor + (1 - fill_factor) * gap_transmission
    effective_white = max_transmission * fill_factor + (1 - fill_factor) * gap_transmission
    effective_contrast = effective_white / effective_black if effective_black > 0 else float('inf')
    
    print(f"  Commanded BLACK → Actual: {effective_black:.3f}")
    print(f"  Commanded WHITE → Actual: {effective_white:.3f}")
    print(f"  Effective contrast: {effective_contrast:.1f}:1")
    
    if fill_factor < 1.0:
        print(f"\n⚠ Impact of {(1-fill_factor)*100:.1f}% gap area:")
        print(f"  • Blacks are {effective_black/min_transmission:.2f}× brighter (gap leakage)")
        print(f"  • Whites are {effective_white/max_transmission:.2f}× dimmer (gap blocks light)")
        print(f"  • Effective contrast reduced from {contrast_ratio:.1f}:1 to {effective_contrast:.1f}:1")
    
    # Save
    output_file = 'lcd_calibration.json'
    save_calibration(fill_factor, contrast_ratio, min_transmission, max_transmission,
                     gap_transmission, output_file)
    
    # Next steps
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print(f"\n✓ Calibration complete!")
    print(f"\n1. Review saved file: {output_file}")
    print(f"2. Run: python optical_classifier_amplitude_REALISTIC.py")
    print(f"3. The simulation will automatically use your measured values")
    print(f"4. Compare IDEAL vs CALIBRATED accuracy")
    print(f"\n💡 To recalibrate, just run this script again.")
    print("="*70 + "\n")
    
    # Display saved file content
    print(f"📄 {output_file} contents:")
    print("-" * 70)
    with open(output_file, 'r') as f:
        print(f.read())
    print("-" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Calibration cancelled by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
