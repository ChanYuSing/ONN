"""
Optical Neural Network Configuration

This file contains ALL tunable parameters for the multi-layer amplitude ONN.
Edit this file to configure your optical system and training.

=== SYSTEM OVERVIEW ===

    Light Source → [LCD Layer 1] → gap → [LCD Layer 2] → ... → [LCD Layer N] → Camera
                   (amplitude)    (diffraction)  (amplitude)      (diffraction)   (detector)

Each LCD layer:
    - Controls light transmission per pixel (amplitude modulation)
    - Values in [0, 1]: 0 = block light, 1 = pass light
    - Grayscale support depends on hardware (may be binary only)

Light propagation between layers:
    - Angular spectrum method (Fresnel diffraction)
    - Diffraction spreads light based on wavelength and distance

Camera detection:
    - Measures light intensity |E|² at each pixel
    - Split into zones for classification

=== HARDWARE YOU HAVE ===

Your LCD: 1.54" Monochrome Reflective LCD
    - Resolution: 200×200 pixels
    - Active Area: 27.66mm × 27.66mm
    - Pixel Pitch: 138.3 μm (27.66mm / 200 = 0.1383mm)
    - Type: Monochrome (grayscale TBD)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


# =============================================================================
# 1. HARDWARE CONFIGURATION
# =============================================================================

@dataclass
class LCDConfig:
    """
    Physical specifications of your LCD screen.
    
    IMPORTANT: These values directly affect physics simulation accuracy!
    Measure your actual hardware if possible.
    """
    # --- Known from datasheet ---
    resolution: Tuple[int, int] = (200, 200)  # pixels (width, height)
    active_area: Tuple[float, float] = (27.66e-3, 27.66e-3)  # meters (27.66mm)
    pixel_pitch: float = 138.3e-6  # meters (138.3 μm, calculated from active_area/resolution)
    
    # --- Measured or estimated (run calibration to update) ---
    fill_factor: float = 1.0  # Ratio of active pixel area (1.0 = ideal, typical: 0.7-0.9)
    gap_transmission: float = 0.5  # Light leakage through pixel gaps (0-1)
    min_transmission: float = 0.0  # Black pixel transmission (ideal: 0)
    max_transmission: float = 1.0  # White pixel transmission (ideal: 1)
    contrast_ratio: float = float('inf')  # Max/min transmission ratio
    
    # --- Grayscale capability ---
    grayscale_levels: int = 256  # Number of gray levels (2 = binary, 256 = 8-bit)
    is_binary: bool = False  # If True, only use 0 or 1 (no grayscale)
    
    @property
    def width_pixels(self) -> int:
        return self.resolution[0]
    
    @property
    def height_pixels(self) -> int:
        return self.resolution[1]


@dataclass
class LightSourceConfig:
    """
    Light source specifications.
    
    Common options:
    - Green laser: 532nm (recommended for coherent light)
    - Red laser: 632.8nm (HeNe) or 650nm (cheap diode)
    - LED: Various wavelengths (partially coherent)
    """
    wavelength: float = 532e-9  # meters (532nm green laser)
    coherence_length: float = 0.1  # meters (10cm typical for cheap laser)
    is_coherent: bool = True  # True for laser, False for LED
    
    # Beam properties
    beam_diameter: float = 5e-3  # meters (5mm typical collimated beam)
    beam_uniformity: float = 0.9  # 0-1, how uniform is the illumination


@dataclass
class CameraConfig:
    """
    Camera/detector specifications.
    
    Options:
    - Simple photodetectors (just measure total intensity per zone)
    - Camera (measure full 2D intensity pattern)
    """
    resolution: Tuple[int, int] = (200, 200)  # Should match or exceed LCD resolution
    pixel_size: float = 138.3e-6  # meters (ideally match LCD pixel pitch)
    bit_depth: int = 8  # bits per pixel (8 = 256 levels, 12 = 4096 levels)
    noise_level: float = 0.01  # Relative noise (0.01 = 1% of max signal)


# =============================================================================
# 2. MULTI-LAYER SYSTEM CONFIGURATION  
# =============================================================================

@dataclass
class MultiLayerConfig:
    """
    Configuration for cascaded LCD layers.
    
    Key insight: More layers = more expressive power, but harder to align physically.
    
    Example setups:
    - 1 layer: Simplest, limited expressiveness
    - 2 layers: Good balance of power and practicality
    - 3+ layers: More powerful but alignment becomes critical
    """
    # --- Number and spacing of layers ---
    num_layers: int = 2  # Number of LCD screens
    
    # Layer spacings (distance between consecutive layers in meters)
    # len(layer_spacings) should equal num_layers (last one is to camera)
    layer_spacings: List[float] = field(default_factory=lambda: [5e-2, 5e-2])  # 5cm each
    
    # --- Physical constraints ---
    min_spacing: float = 1e-2  # Minimum practical spacing (1cm)
    max_spacing: float = 20e-2  # Maximum before coherence issues (20cm)
    
    # --- Alignment tolerances (for realistic simulation) ---
    lateral_alignment_error: float = 0.0  # meters (0 = perfect alignment)
    rotation_alignment_error: float = 0.0  # radians (0 = no rotation)
    spacing_tolerance: float = 0.0  # meters (uncertainty in spacing)
    
    # --- Per-layer transmission loss ---
    # Each layer absorbs/reflects some light even in "transparent" state
    per_layer_transmission: float = 0.95  # 95% transmitted per layer
    
    def __post_init__(self):
        # Validate spacings
        if len(self.layer_spacings) != self.num_layers:
            # Default: equal spacing
            self.layer_spacings = [5e-2] * self.num_layers
    
    @property
    def total_path_length(self) -> float:
        """Total optical path from first layer to camera."""
        return sum(self.layer_spacings)


# =============================================================================
# 3. INPUT CONFIGURATION
# =============================================================================

@dataclass
class InputConfig:
    """
    How to prepare input images for the optical system.
    """
    # --- MNIST preprocessing ---
    original_size: Tuple[int, int] = (28, 28)  # MNIST original size
    digit_size: int = 140  # Resize digit to this size (28×5=140), then pad to LCD size
    
    # How to fit input onto LCD
    # Options: 'center_pad', 'resize', 'tile'
    input_mode: str = 'center_pad'  # Zero-pad to LCD size, center the digit
    
    # For center_pad mode
    pad_value: float = 0.0  # Value for padding (0 = black, 1 = white)
    
    # Intensity normalization: rescale each image so total energy is constant
    # Physically simulates consistent laser illumination across all inputs
    normalize_energy: bool = False
    
    # For resize mode (if needed)
    resize_to: Optional[Tuple[int, int]] = None  # Resize before padding
    
    # --- Intensity scaling ---
    # MNIST pixel values are 0-255, need to convert to optical intensity 0-1
    normalize: bool = True  # Normalize to [0, 1]
    intensity_scale: float = 1.0  # Multiply input intensity by this factor
    
    # --- Input encoding type ---
    # 'amplitude': Input image modulates light amplitude (standard)
    # 'intensity': Input image IS the light intensity
    encoding: str = 'amplitude'


# =============================================================================
# 4. DETECTION / OUTPUT CONFIGURATION
# =============================================================================

@dataclass 
class DetectionConfig:
    """
    How to detect and classify the output light pattern.
    
    Key decision: Zones vs Full camera
    
    Zones (simpler, cheaper):
        - Split detector into N×N regions
        - Measure total intensity in each region
        - Use zone intensities as features for classification
        - Hardware: Can use N² photodetectors instead of camera
    
    Full camera (more powerful):
        - Capture entire 2D intensity pattern
        - Learn detection masks (where to look)
        - More parameters but higher accuracy
    """
    # --- Zone-based detection ---
    use_zones: bool = True  # If True, split into zones; if False, use full camera
    num_zones_per_side: int = 4  # N×N grid (4 = 16 zones for 10-class problem)
    
    # --- For full camera mode ---
    use_detection_masks: bool = False  # Learn spatial detection masks
    num_detection_masks: int = 10  # One per class (for MNIST)
    
    # --- Classification method ---
    # 'nearest_center': Classify by nearest center in zone-space
    # 'cross_entropy': Direct cross-entropy loss on logits
    # 'distance_softmax': Softmax over negative distances to centers
    classification_method: str = 'distance_softmax'
    
    @property
    def num_zones(self) -> int:
        return self.num_zones_per_side ** 2


# =============================================================================
# 5. TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Training hyperparameters.
    """
    # --- Basic training ---
    epochs: int = 150
    batch_size: int = 64
    learning_rate: float = 0.01
    
    # --- Optimizer ---
    optimizer: str = 'adam'  # 'adam', 'sgd', 'adamw'
    weight_decay: float = 0.001
    momentum: float = 0.9  # For SGD
    
    # --- Learning rate schedule ---
    lr_scheduler: str = 'cosine'  # 'none', 'step', 'cosine', 'plateau'
    lr_step_size: int = 30  # For step scheduler
    lr_gamma: float = 0.1  # LR multiplier at each step
    warmup_epochs: int = 10  # Linear LR warmup from ~0 to full LR
    
    # --- Per-parameter learning rates ---
    # Different components may need different learning rates
    amplitude_lr: float = 0.01  # Learning rate for LCD amplitude masks
    center_lr: float = 0.5  # Learning rate for classification centers
    detection_lr: float = 0.05  # Learning rate for detector params (zone logits, classifiers)
    
    # --- Regularization ---
    amplitude_regularization: float = 0.001  # L2 regularization on amplitude masks
    smoothness_regularization: float = 0.01  # Encourage smooth amplitude patterns
    
    # --- Initialization ---
    amplitude_init: str = 'uniform'  # 'uniform', 'normal', 'ones', 'zeros'
    amplitude_init_scale: float = 0.5  # Initial amplitude mean
    
    # --- Calibration ---
    calibration_batches: int = 50  # Batches to use for center calibration
    recalibrate_every: int = 0  # Re-calibrate centers every N epochs (0 = never)


# =============================================================================
# 6. PHYSICS SIMULATION OPTIONS
# =============================================================================

@dataclass
class PhysicsConfig:
    """
    Physics simulation settings.
    
    Trade-off: More realistic = slower but more accurate to real hardware.
    """
    # --- Propagation method ---
    # 'angular_spectrum': Full wave propagation (accurate, handles diffraction)
    # 'fresnel': Fresnel approximation (faster, good for moderate distances)
    # 'geometric': Ray tracing (fastest, ignores diffraction - NOT recommended)
    propagation_method: str = 'angular_spectrum'
    
    # --- Hardware effects to simulate ---
    simulate_fill_factor: bool = False  # Model pixel gaps
    simulate_contrast_limits: bool = False  # Model non-ideal black/white
    simulate_alignment_errors: bool = False  # Model layer misalignment
    simulate_noise: bool = False  # Add detector noise
    
    # --- Binary LCD simulation ---
    # If LCD is binary (on/off only), use differentiable approximation for training
    binarize_amplitude: bool = False  # Force amplitudes to 0 or 1
    binarization_method: str = 'sigmoid'  # 'sigmoid', 'ste' (straight-through estimator)
    binarization_sharpness: float = 10.0  # Higher = sharper sigmoid → more binary-like
    
    # --- Numerical precision ---
    dtype: str = 'float32'  # 'float32' or 'float64'
    use_complex64: bool = True  # Use complex64 for field computations


# =============================================================================
# 7. EXPERIMENT & LOGGING
# =============================================================================

@dataclass
class ExperimentConfig:
    """
    Experiment settings and logging.
    """
    # --- Experiment identification ---
    experiment_name: str = 'multilayer_amplitude_onn'
    run_id: Optional[str] = None  # Auto-generated if None
    
    # --- Paths ---
    data_dir: str = './data'  # Where to store/load MNIST
    output_dir: str = './outputs'  # Where to save results
    checkpoint_dir: str = './checkpoints'  # Where to save model checkpoints
    
    # --- Logging ---
    log_every_n_batches: int = 50
    save_checkpoint_every_n_epochs: int = 10
    save_amplitude_visualizations: bool = True
    save_propagation_visualizations: bool = True
    
    # --- Device ---
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    
    # --- Reproducibility ---
    seed: int = 42
    deterministic: bool = True


# =============================================================================
# 8. MASTER CONFIG (COMBINES ALL)
# =============================================================================

@dataclass
class Config:
    """
    Master configuration that combines all components.
    
    Usage:
        from config import Config
        cfg = Config()
        
        # Access parameters
        cfg.lcd.pixel_pitch  # 138.3e-6
        cfg.multilayer.num_layers  # 2
        cfg.training.learning_rate  # 0.02
        
        # Modify and use
        cfg.multilayer.num_layers = 3
        cfg.multilayer.layer_spacings = [3e-2, 4e-2, 5e-2]
    """
    lcd: LCDConfig = field(default_factory=LCDConfig)
    light: LightSourceConfig = field(default_factory=LightSourceConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    multilayer: MultiLayerConfig = field(default_factory=MultiLayerConfig)
    input: InputConfig = field(default_factory=InputConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def __post_init__(self):
        # Ensure consistency between configs
        self._validate()
    
    def _validate(self):
        """Check for inconsistent configuration."""
        errors = []
        
        # Check coherence length vs total path
        if self.multilayer.total_path_length > self.light.coherence_length:
            errors.append(
                f"Total path ({self.multilayer.total_path_length*100:.1f}cm) exceeds "
                f"coherence length ({self.light.coherence_length*100:.1f}cm)"
            )
        
        # Check layer spacings count
        if len(self.multilayer.layer_spacings) != self.multilayer.num_layers:
            errors.append(
                f"layer_spacings length ({len(self.multilayer.layer_spacings)}) "
                f"!= num_layers ({self.multilayer.num_layers})"
            )
        
        # Check zones vs classes
        if self.detection.use_zones and self.detection.num_zones < 10:
            print(f"WARNING: Only {self.detection.num_zones} zones for 10 classes")
        
        if errors:
            for e in errors:
                print(f"CONFIG ERROR: {e}")
    
    @property
    def N(self) -> int:
        """Grid size for simulation (LCD resolution)."""
        return self.lcd.resolution[0]
    
    @property
    def dx(self) -> float:
        """Pixel pitch (alias for physics calculations)."""
        return self.lcd.pixel_pitch
    
    @property
    def wavelength(self) -> float:
        """Light wavelength (alias for physics calculations)."""
        return self.light.wavelength
    
    def get_device(self):
        """Get torch device based on configuration."""
        import torch
        if self.experiment.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.experiment.device)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        from dataclasses import asdict
        return asdict(self)
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        import json
        with open(filepath, 'w') as f:
            # Handle special values like inf
            d = self.to_dict()
            json.dump(d, f, indent=2, default=lambda x: str(x) if x == float('inf') else x)
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        import json
        with open(filepath, 'r') as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items()})
    
    def summary(self):
        """Print configuration summary."""
        print("=" * 70)
        print("OPTICAL NEURAL NETWORK CONFIGURATION")
        print("=" * 70)
        
        print(f"\n📺 LCD Hardware:")
        print(f"   Resolution: {self.lcd.resolution[0]}×{self.lcd.resolution[1]}")
        print(f"   Pixel pitch: {self.lcd.pixel_pitch*1e6:.1f} μm")
        print(f"   Grayscale: {'Binary' if self.lcd.is_binary else f'{self.lcd.grayscale_levels} levels'}")
        
        print(f"\n💡 Light Source:")
        print(f"   Wavelength: {self.light.wavelength*1e9:.0f} nm")
        print(f"   Coherence: {'Coherent' if self.light.is_coherent else 'Incoherent'}")
        
        print(f"\n🔄 Multi-Layer System:")
        print(f"   Layers: {self.multilayer.num_layers}")
        print(f"   Spacings: {[f'{s*100:.1f}cm' for s in self.multilayer.layer_spacings]}")
        print(f"   Total path: {self.multilayer.total_path_length*100:.1f} cm")
        
        print(f"\n📷 Detection:")
        print(f"   Method: {'Zone-based' if self.detection.use_zones else 'Full camera'}")
        if self.detection.use_zones:
            print(f"   Zones: {self.detection.num_zones_per_side}×{self.detection.num_zones_per_side} = {self.detection.num_zones}")
        
        print(f"\n🎯 Training:")
        print(f"   Epochs: {self.training.epochs}")
        print(f"   Batch size: {self.training.batch_size}")
        print(f"   Learning rate: {self.training.learning_rate}")
        
        total_params = self.multilayer.num_layers * self.lcd.resolution[0] * self.lcd.resolution[1]
        print(f"\n📊 Learnable Parameters:")
        print(f"   Amplitude masks: {self.multilayer.num_layers} × {self.N}×{self.N} = {total_params:,}")
        
        print("=" * 70)


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def get_simple_config() -> Config:
    """Simple 1-layer config for testing."""
    cfg = Config()
    cfg.multilayer.num_layers = 1
    cfg.multilayer.layer_spacings = [5e-2]  # 5cm to camera
    cfg.detection.num_zones_per_side = 4
    cfg.training.epochs = 50
    return cfg


def get_2layer_config() -> Config:
    """Standard 2-layer configuration."""
    cfg = Config()
    cfg.multilayer.num_layers = 2
    cfg.multilayer.layer_spacings = [3e-2, 5e-2]  # 3cm between layers, 5cm to camera
    cfg.detection.num_zones_per_side = 4
    cfg.training.epochs = 100
    return cfg


def get_3layer_config() -> Config:
    """3-layer configuration for higher expressiveness."""
    cfg = Config()
    cfg.multilayer.num_layers = 3
    cfg.multilayer.layer_spacings = [2e-2, 3e-2, 5e-2]
    cfg.detection.num_zones_per_side = 4
    cfg.training.epochs = 150
    return cfg


def get_binary_lcd_config() -> Config:
    """Configuration for binary (on/off only) LCD."""
    cfg = Config()
    cfg.lcd.is_binary = True
    cfg.lcd.grayscale_levels = 2
    cfg.physics.binarize_amplitude = True
    cfg.physics.binarization_sharpness = 10.0
    return cfg


def get_realistic_hardware_config() -> Config:
    """Configuration with realistic hardware effects."""
    cfg = Config()
    cfg.lcd.fill_factor = 0.85  # 85% fill factor
    cfg.lcd.min_transmission = 0.05  # 5% leakage in black
    cfg.lcd.max_transmission = 0.90  # 90% max transmission
    cfg.physics.simulate_fill_factor = True
    cfg.physics.simulate_contrast_limits = True
    cfg.physics.simulate_noise = True
    return cfg


# =============================================================================
# DEFAULT INSTANCE
# =============================================================================

# Create default configuration instance
# Import and modify in your scripts:
#   from config import cfg
#   cfg.multilayer.num_layers = 3

cfg = Config()


# =============================================================================
# MAIN: Show config summary when run directly
# =============================================================================

if __name__ == '__main__':
    print("\nDefault Configuration:")
    cfg.summary()
    
    print("\n\n2-Layer Preset:")
    get_2layer_config().summary()
