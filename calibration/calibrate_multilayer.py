"""
Multi-Layer LCD Calibration Tool

Purpose: Measure inter-layer parameters for optical_classifier_multilayer.py

What This Tool Does:
1. Guides you through measuring physical spacings between layers
2. Calibrates alignment errors (X-Y position, rotation)
3. Measures cumulative transmission loss through multiple layers
4. Checks coherence length requirements
5. Generates multilayer_calibration.json

Equipment Needed:
- Multiple LCD screens (already mounted in cascade)
- Ruler or digital caliper (for spacing measurement)
- Photodetector or light meter (for transmission loss)
- Alignment jig or translation stages (for alignment calibration)
- Green laser pointer (532nm)

Measurements to Perform:
1. Layer Spacing: Physical distance between each consecutive pair
2. Alignment Errors: X-Y lateral shifts and rotational misalignment
3. Transmission Loss: Intensity loss through each layer
4. Coherence Check: Verify laser coherence length is sufficient

Output: multilayer_calibration.json (used by optical_classifier_multilayer.py)

Philosophy: 
  - Start with IDEAL assumptions (perfect alignment, no loss)
  - Measure REAL hardware with simple tools
  - Update parameters to match reality
"""

import numpy as np
import json
import time
from pathlib import Path

# ============================================================
# SINGLE LCD SPECIFICATIONS (from main script)
# ============================================================

LCD_SPECS = {
    'resolution': (200, 200),
    'active_area': (27.66e-3, 27.66e-3),  # 27.66mm
    'dot_pitch': 138.3e-6,  # 138.3 μm
}

print("="*70)
print("MULTI-LAYER LCD CALIBRATION TOOL")
print("="*70)
print("\nThis tool will guide you through measuring inter-layer parameters.")
print(f"\nSingle LCD Specs:")
print(f"  Resolution: {LCD_SPECS['resolution'][0]}×{LCD_SPECS['resolution'][1]} pixels")
print(f"  Pixel pitch: {LCD_SPECS['dot_pitch']*1e6:.1f} μm")
print(f"  Active area: {LCD_SPECS['active_area'][0]*1e3:.2f}mm × {LCD_SPECS['active_area'][1]*1e3:.2f}mm")

print(f"\nPhilosophy:")
print(f"  • Start: IDEAL (perfect alignment, no loss)")
print(f"  • Measure: REAL hardware with simple tools")
print(f"  • Update: Parameters to match reality")

# ============================================================
# STEP 1: SYSTEM CONFIGURATION
# ============================================================

def configure_system():
    """
    Ask user about system configuration (number of layers).
    """
    print("\n" + "="*70)
    print("STEP 1: SYSTEM CONFIGURATION")
    print("="*70)
    
    print("\nHow many LCD screens are in your cascaded system?")
    print("  Examples:")
    print("    2 layers: Light → LCD1 → gap → LCD2 → Detector")
    print("    3 layers: Light → LCD1 → gap → LCD2 → gap → LCD3 → Detector")
    
    while True:
        try:
            num_layers = int(input("\nEnter number of layers [1-10]: ").strip())
            if 1 <= num_layers <= 10:
                break
            print("⚠ Invalid. Must be between 1 and 10.")
        except ValueError:
            print("⚠ Invalid input. Please enter a number.")
    
    print(f"\n✓ System has {num_layers} layers")
    print(f"  Number of gaps to measure: {num_layers - 1}")
    
    return num_layers


# ============================================================
# STEP 2: LAYER SPACING MEASUREMENT
# ============================================================

def measure_layer_spacings(num_layers):
    """
    Measure the physical distance between consecutive layers.
    
    Methods:
    1. RULER/CALIPER: Direct measurement with ruler or digital caliper
    2. NOMINAL: Use designed spacing (ideal assumption)
    3. ESTIMATE: Use typical values
    """
    print("\n" + "="*70)
    print("STEP 2: LAYER SPACING MEASUREMENT")
    print("="*70)
    
    print("\nWhat is Layer Spacing?")
    print("  The physical distance between consecutive LCD screens")
    print("  Example: If LCD1 and LCD2 are separated by 15cm spacer → spacing = 0.15m")
    
    print("\nWhy This Matters:")
    print("  • Spacing controls diffraction strength")
    print("  • Optimal: 10-20cm for 532nm green light")
    print("  • Too small: Not enough diffraction (layers don't interact)")
    print("  • Too large: Over-diffraction (blurry output)")
    
    print("\nMeasurement Methods:")
    print("  [1] RULER/CALIPER: Direct measurement (RECOMMENDED)")
    print("  [2] NOMINAL: Use designed spacing (if following build plans)")
    print("  [3] ESTIMATE: Use typical values")
    
    layer_spacings = []
    
    for gap_idx in range(num_layers - 1):
        print(f"\n--- Gap {gap_idx + 1} (between Layer {gap_idx + 1} and Layer {gap_idx + 2}) ---")
        
        while True:
            choice = input(f"Select method [1/2/3]: ").strip()
            
            if choice == '1':
                print(f"\n📏 RULER/CALIPER METHOD:")
                print(f"\nMeasure the distance between:")
                print(f"  • Front surface of Layer {gap_idx + 1}")
                print(f"  • Front surface of Layer {gap_idx + 2}")
                print(f"\nTips:")
                print(f"  • Measure from LCD surface to LCD surface (not including mounts)")
                print(f"  • Measure at multiple points and average")
                print(f"  • Use digital caliper for accuracy (<1mm error)")
                
                spacing_cm = float(input(f"\nEnter measured spacing (cm): "))
                spacing_m = spacing_cm / 100.0
                
                print(f"✓ Gap {gap_idx + 1} spacing: {spacing_cm:.1f} cm = {spacing_m:.3f} m")
                layer_spacings.append(spacing_m)
                break
            
            elif choice == '2':
                print(f"\n📐 NOMINAL METHOD:")
                print(f"\nWhat is your designed spacing for Gap {gap_idx + 1}?")
                print(f"  Examples:")
                print(f"    • 10 cm (0.10 m) - Close spacing")
                print(f"    • 15 cm (0.15 m) - Recommended spacing")
                print(f"    • 20 cm (0.20 m) - Wide spacing")
                
                spacing_cm = float(input(f"\nEnter nominal spacing (cm): "))
                spacing_m = spacing_cm / 100.0
                
                print(f"✓ Gap {gap_idx + 1} spacing: {spacing_cm:.1f} cm = {spacing_m:.3f} m")
                layer_spacings.append(spacing_m)
                break
            
            elif choice == '3':
                print(f"\n📊 ESTIMATE METHOD:")
                print(f"\nTypical spacings:")
                print(f"  • 10 cm: Compact systems, tight budget")
                print(f"  • 15 cm: Recommended (optimal diffraction)")
                print(f"  • 20 cm: Research systems, high accuracy")
                
                spacing_cm = float(input(f"\nEnter estimated spacing (cm) [10-50]: "))
                spacing_m = spacing_cm / 100.0
                
                print(f"✓ Gap {gap_idx + 1} spacing: {spacing_cm:.1f} cm = {spacing_m:.3f} m")
                layer_spacings.append(spacing_m)
                break
            
            else:
                print("⚠ Invalid choice. Please enter 1, 2, or 3.")
    
    # Summary
    total_path = sum(layer_spacings)
    
    print(f"\n✓ All spacings measured:")
    for i, spacing in enumerate(layer_spacings):
        print(f"  Gap {i + 1}: {spacing*100:.1f} cm")
    print(f"\nTotal optical path: {total_path*100:.1f} cm")
    
    return layer_spacings


# ============================================================
# STEP 3: ALIGNMENT ERROR MEASUREMENT
# ============================================================

def measure_alignment_errors(num_layers):
    """
    Measure alignment errors (lateral shifts and rotation) for each layer.
    
    Methods:
    1. PRECISION MEASUREMENT: Use alignment targets + microscope
    2. VISUAL ESTIMATION: Estimate by eye
    3. SKIP: Assume perfect alignment (IDEAL)
    """
    print("\n" + "="*70)
    print("STEP 3: ALIGNMENT ERROR MEASUREMENT")
    print("="*70)
    
    print("\nWhat are Alignment Errors?")
    print("  Real hardware has imperfect positioning:")
    print("  • Lateral (X-Y): LCD shifted horizontally")
    print("  • Rotation (θ): LCD rotated around center")
    print("  • Spacing error: Actual spacing ≠ nominal spacing")
    
    print("\nWhy This Matters:")
    print("  • Small errors (<100 μm): Minimal impact on accuracy")
    print("  • Medium errors (100-500 μm): ~5% accuracy loss")
    print("  • Large errors (>500 μm): >10% accuracy loss")
    
    print("\nMeasurement Methods:")
    print("  [1] PRECISION: Use alignment targets (RECOMMENDED if available)")
    print("  [2] ESTIMATE: Typical values based on mounting quality")
    print("  [3] SKIP: Assume perfect alignment (IDEAL)")
    
    while True:
        choice = input("\nSelect method [1/2/3]: ").strip()
        
        if choice == '1':
            print(f"\n🎯 PRECISION METHOD:")
            print(f"\nSetup:")
            print(f"  1. Display alignment crosshair on all layers")
            print(f"  2. View through stack with microscope or camera")
            print(f"  3. Measure X-Y shift and rotation of each layer relative to Layer 1")
            
            lateral_errors = [[0.0, 0.0]]  # Layer 1 is reference (no error)
            rotation_errors = [0.0]  # Layer 1 is reference
            spacing_errors = [0.0]  # Layer 1 has no spacing error
            
            for layer_idx in range(1, num_layers):
                print(f"\n--- Layer {layer_idx + 1} (relative to Layer 1) ---")
                
                shift_x_um = float(input(f"Enter X shift (μm) [+ = right, - = left]: "))
                shift_y_um = float(input(f"Enter Y shift (μm) [+ = up, - = down]: "))
                shift_x_m = shift_x_um * 1e-6
                shift_y_m = shift_y_um * 1e-6
                
                rotation_deg = float(input(f"Enter rotation (degrees) [+ = CCW, - = CW]: "))
                
                spacing_error_mm = float(input(f"Enter spacing error (mm) [+ = farther, - = closer]: "))
                spacing_error_m = spacing_error_mm * 1e-3
                
                lateral_errors.append([shift_x_m, shift_y_m])
                rotation_errors.append(rotation_deg)
                spacing_errors.append(spacing_error_m)
                
                print(f"✓ Layer {layer_idx + 1}: Δx={shift_x_um:.1f}μm, "
                      f"Δy={shift_y_um:.1f}μm, Δθ={rotation_deg:.2f}°, "
                      f"Δz={spacing_error_mm:.2f}mm")
            
            return {
                'lateral': lateral_errors,
                'rotation': rotation_errors,
                'spacing_error': spacing_errors
            }
        
        elif choice == '2':
            print(f"\n📊 ESTIMATE METHOD:")
            print(f"\nTypical alignment errors by mounting quality:")
            print(f"  High-quality (XYZ stages): <50 μm lateral, <0.2° rotation")
            print(f"  Medium-quality (manual): 50-200 μm lateral, 0.2-0.5° rotation")
            print(f"  Low-quality (3D printed): 200-500 μm lateral, 0.5-2° rotation")
            
            quality = input(f"\nEnter mounting quality [high/medium/low]: ").strip().lower()
            
            if quality == 'high':
                lateral_std = 30e-6  # 30 μm
                rotation_std = 0.15  # 0.15°
            elif quality == 'medium':
                lateral_std = 100e-6  # 100 μm
                rotation_std = 0.35  # 0.35°
            else:  # low
                lateral_std = 300e-6  # 300 μm
                rotation_std = 1.0  # 1°
            
            # Generate random errors (Gaussian)
            np.random.seed(42)  # Reproducible
            lateral_errors = [[0.0, 0.0]]  # Layer 1 is reference
            rotation_errors = [0.0]
            spacing_errors = [0.0]
            
            for layer_idx in range(1, num_layers):
                shift_x = np.random.randn() * lateral_std
                shift_y = np.random.randn() * lateral_std
                rotation = np.random.randn() * rotation_std
                spacing_err = np.random.randn() * 1e-3  # 1mm std
                
                lateral_errors.append([shift_x, shift_y])
                rotation_errors.append(rotation)
                spacing_errors.append(spacing_err)
                
                print(f"✓ Layer {layer_idx + 1}: Δx={shift_x*1e6:.1f}μm, "
                      f"Δy={shift_y*1e6:.1f}μm, Δθ={rotation:.2f}°")
            
            return {
                'lateral': lateral_errors,
                'rotation': rotation_errors,
                'spacing_error': spacing_errors
            }
        
        elif choice == '3':
            print(f"\n⚠ SKIPPING: Assuming perfect alignment (IDEAL)")
            
            lateral_errors = [[0.0, 0.0] for _ in range(num_layers)]
            rotation_errors = [0.0 for _ in range(num_layers)]
            spacing_errors = [0.0 for _ in range(num_layers)]
            
            return {
                'lateral': lateral_errors,
                'rotation': rotation_errors,
                'spacing_error': spacing_errors
            }
        
        else:
            print("⚠ Invalid choice. Please enter 1, 2, or 3.")


# ============================================================
# STEP 4: TRANSMISSION LOSS MEASUREMENT
# ============================================================

def measure_transmission_loss(num_layers):
    """
    Measure transmission loss through each layer.
    
    Methods:
    1. PHOTODETECTOR: Measure intensity before/after each layer
    2. ESTIMATE: Use typical values (70% per layer)
    3. SKIP: Assume ideal (100% transmission)
    """
    print("\n" + "="*70)
    print("STEP 4: TRANSMISSION LOSS MEASUREMENT")
    print("="*70)
    
    print("\nWhat is Transmission Loss?")
    print("  Each LCD absorbs/reflects some light:")
    print("  • Typical monochrome LCD: 70% transmission")
    print("  • High-quality AR-coated: 90-95% transmission")
    print("  • Cumulative effect: T_total = T1 × T2 × T3 × ...")
    
    print("\nWhy This Matters:")
    print("  • 2 layers @ 70%: 49% total (half lost!)")
    print("  • 3 layers @ 70%: 34% total (two-thirds lost!)")
    print("  • Solution: Brighter light source or AR coatings")
    
    print("\nMeasurement Methods:")
    print("  [1] PHOTODETECTOR: Measure intensity (RECOMMENDED)")
    print("  [2] ESTIMATE: Use typical values")
    print("  [3] SKIP: Assume ideal (100% transmission)")
    
    while True:
        choice = input("\nSelect method [1/2/3]: ").strip()
        
        if choice == '1':
            print(f"\n📊 PHOTODETECTOR METHOD:")
            print(f"\nSetup:")
            print(f"  • Laser → [No LCDs] → Photodetector")
            print(f"  • Measure I0 (baseline intensity)")
            print(f"\nThen sequentially add each layer:")
            print(f"  • Laser → [LCD1] → Photodetector → measure I1")
            print(f"  • Laser → [LCD1] → [LCD2] → Photodetector → measure I2")
            print(f"  • Continue for all layers...")
            
            input("\n[Press Enter when ready to measure I0 (no LCDs)]")
            I0 = float(input("Enter I0 (baseline intensity, any units): "))
            
            layer_transmissions = []
            I_prev = I0
            
            for layer_idx in range(num_layers):
                input(f"\n[Press Enter when Layer {layer_idx + 1} is inserted]")
                I_current = float(input(f"Enter intensity after Layer {layer_idx + 1} (same units): "))
                
                # Transmission of this specific layer
                T_layer = I_current / I_prev
                layer_transmissions.append(T_layer)
                
                print(f"✓ Layer {layer_idx + 1} transmission: {T_layer:.3f} ({T_layer*100:.1f}%)")
                print(f"  Cumulative transmission: {I_current/I0:.3f} ({I_current/I0*100:.1f}%)")
                
                I_prev = I_current
            
            return layer_transmissions
        
        elif choice == '2':
            print(f"\n📊 ESTIMATE METHOD:")
            print(f"\nTypical transmission by LCD type:")
            print(f"  Basic monochrome: 0.60 - 0.75")
            print(f"  Standard monochrome: 0.70 - 0.80")
            print(f"  High-quality: 0.80 - 0.90")
            print(f"  AR-coated: 0.90 - 0.95")
            
            transmission_single = float(input(f"\nEnter transmission per layer [0.5-0.95]: "))
            
            layer_transmissions = [transmission_single] * num_layers
            
            cumulative = transmission_single ** num_layers
            print(f"\n✓ Transmission per layer: {transmission_single:.3f}")
            print(f"  Cumulative ({num_layers} layers): {cumulative:.3f} ({cumulative*100:.1f}%)")
            
            return layer_transmissions
        
        elif choice == '3':
            print(f"\n⚠ SKIPPING: Assuming ideal (100% transmission)")
            return [1.0] * num_layers
        
        else:
            print("⚠ Invalid choice. Please enter 1, 2, or 3.")


# ============================================================
# STEP 5: COHERENCE LENGTH CHECK
# ============================================================

def check_coherence_length(total_path_length):
    """
    Check if light source has sufficient coherence length.
    
    Provide guidance on required laser type.
    """
    print("\n" + "="*70)
    print("STEP 5: COHERENCE LENGTH CHECK")
    print("="*70)
    
    print("\nWhat is Coherence Length?")
    print("  How far light can travel and still interfere")
    print("  • LED: 10-50 μm (NOT suitable for multi-layer!)")
    print("  • Laser diode (multimode): 1-10 mm")
    print("  • Laser diode (single-mode): 10-100 cm")
    print("  • HeNe laser: 10-100 m")
    
    print(f"\nYour System:")
    print(f"  Total optical path: {total_path_length*100:.1f} cm")
    print(f"  Required coherence length: >{total_path_length*100:.1f} cm")
    
    print(f"\nRecommended Light Source:")
    if total_path_length < 0.01:  # < 1cm
        print(f"  ✓ LED or any laser OK")
        recommended_source = "LED or laser diode"
        coherence = 0.05  # 5 cm (conservative for LED)
    elif total_path_length < 0.1:  # < 10cm
        print(f"  ✓ Laser diode (multimode) OK ($20-50)")
        recommended_source = "Laser diode (multimode)"
        coherence = 0.10  # 10 cm
    elif total_path_length < 0.5:  # < 50cm
        print(f"  ✓ Laser diode (single-mode) OK ($50-100)")
        recommended_source = "Laser diode (single-mode)"
        coherence = 0.50  # 50 cm
    else:  # > 50cm
        print(f"  ⚠ HeNe laser or high-quality single-mode required ($200+)")
        recommended_source = "HeNe laser"
        coherence = 10.0  # 10 m
    
    print(f"\nDo you have a different light source?")
    custom = input(f"Use recommended ({recommended_source})? [y/n]: ").strip().lower()
    
    if custom == 'n':
        print(f"\nEnter your light source coherence length:")
        coherence = float(input(f"Coherence length (cm): ")) / 100.0
    
    if coherence < total_path_length:
        print(f"\n⚠ WARNING: Coherence length ({coherence*100:.1f}cm) < "
              f"Total path ({total_path_length*100:.1f}cm)")
        print(f"  Interference will be DEGRADED!")
        print(f"  Consider: Shorter spacings or better laser")
    else:
        print(f"\n✓ Coherence length sufficient ({coherence*100:.1f}cm > "
              f"{total_path_length*100:.1f}cm)")
    
    return coherence


# ============================================================
# SAVE CALIBRATION
# ============================================================

def save_multilayer_calibration(num_layers, layer_spacings, alignment_errors,
                                 layer_transmissions, coherence_length,
                                 output_file='multilayer_calibration.json'):
    """
    Save multi-layer calibration data to JSON file.
    """
    calibration = {
        'num_layers': num_layers,
        'layer_spacings': layer_spacings,
        'alignment_errors': alignment_errors,
        'layer_transmissions': layer_transmissions,
        'coherence_length': coherence_length,
        'calibrated': True,
        'total_optical_path': sum(layer_spacings),
        'cumulative_transmission': np.prod(layer_transmissions),
    }
    
    with open(output_file, 'w') as f:
        json.dump(calibration, f, indent=2)
    
    print(f"\n✓ Multi-layer calibration saved to {output_file}")
    return calibration


# ============================================================
# MAIN CALIBRATION WORKFLOW
# ============================================================

def main():
    """
    Main multi-layer calibration workflow.
    """
    
    # Step 1: System configuration
    num_layers = configure_system()
    
    # Step 2: Layer spacing
    layer_spacings = measure_layer_spacings(num_layers)
    total_path = sum(layer_spacings)
    
    # Step 3: Alignment errors
    alignment_errors = measure_alignment_errors(num_layers)
    
    # Step 4: Transmission loss
    layer_transmissions = measure_transmission_loss(num_layers)
    cumulative_transmission = np.prod(layer_transmissions)
    
    # Step 5: Coherence check
    coherence_length = check_coherence_length(total_path)
    
    # Summary
    print("\n" + "="*70)
    print("MULTI-LAYER CALIBRATION SUMMARY")
    print("="*70)
    
    print(f"\nSystem Configuration:")
    print(f"  Number of layers: {num_layers}")
    print(f"  Total optical path: {total_path*100:.1f} cm")
    
    print(f"\nLayer Spacings:")
    for i, spacing in enumerate(layer_spacings):
        print(f"  Gap {i + 1}: {spacing*100:.1f} cm")
    
    print(f"\nAlignment Errors:")
    for i in range(num_layers):
        lateral = alignment_errors['lateral'][i]
        rotation = alignment_errors['rotation'][i]
        print(f"  Layer {i + 1}: Δx={lateral[0]*1e6:.1f}μm, "
              f"Δy={lateral[1]*1e6:.1f}μm, Δθ={rotation:.2f}°")
    
    print(f"\nTransmission Loss:")
    for i, trans in enumerate(layer_transmissions):
        print(f"  Layer {i + 1}: {trans:.3f} ({trans*100:.1f}%)")
    print(f"  Cumulative: {cumulative_transmission:.3f} ({cumulative_transmission*100:.1f}%)")
    
    print(f"\nCoherence Length:")
    print(f"  Required: >{total_path*100:.1f} cm")
    print(f"  Measured: {coherence_length*100:.1f} cm")
    if coherence_length >= total_path:
        print(f"  ✓ Sufficient")
    else:
        print(f"  ⚠ WARNING: Insufficient!")
    
    # Save
    output_file = 'multilayer_calibration.json'
    save_multilayer_calibration(num_layers, layer_spacings, alignment_errors,
                                 layer_transmissions, coherence_length, output_file)
    
    # Next steps
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print(f"\n✓ Multi-layer calibration complete!")
    print(f"\n1. Review saved file: {output_file}")
    print(f"2. Run: python optical_classifier_multilayer.py")
    print(f"3. The simulation will use your measured multi-layer parameters")
    print(f"4. Compare accuracy: 1-layer (IDEAL) vs {num_layers}-layer (CALIBRATED)")
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
