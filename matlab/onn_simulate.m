%% onn_simulate.m
%  Optical Neural Network — MATLAB Independent Validation
%  ======================================================
%  Simulates 2-layer amplitude-mask diffractive ONN using ASM.
%  Compares with Python results to validate physics implementation.
%
%  Prerequisites:
%    Run "python export_matlab.py" first to generate .mat files.
%
%  References:
%    Goodman, "Introduction to Fourier Optics", Ch. 3-4 (ASM method)

clear; clc; close all;

%% ========== 1. LOAD DATA ==========
fprintf('Loading data...\n');
load('matlab_export/params.mat');     % wavelength, pixel_pitch, gap, N
load('matlab_export/masks.mat');      % mask1, mask2 [200x200]
load('matlab_export/digits.mat');     % digit0_sample0, digit1_sample0, ...
load('matlab_export/zone_map.mat');   % zone_to_digit [25x1], grid_size

% Ensure all params are double
wavelength = double(wavelength);
pixel_pitch = double(pixel_pitch);
gap = double(gap);
N = double(N);
grid_size = double(grid_size);
% scipy.io.savemat preserves numpy array orientation → no transpose needed
mask1 = double(mask1);
mask2 = double(mask2);

fprintf('  Wavelength:   %.1f nm\n', wavelength*1e9);
fprintf('  Pixel pitch:  %.1f um\n', pixel_pitch*1e6);
fprintf('  Gap:          %.1f mm\n', gap*1e3);
fprintf('  Grid:         %d x %d\n', N, N);
fprintf('  Masks loaded: mask1 [%.4f, %.4f], mask2 [%.4f, %.4f]\n', ...
    min(mask1(:)), max(mask1(:)), min(mask2(:)), max(mask2(:)));

%% ========== 2. LOAD TRANSFER FUNCTION (ASM) ==========
% IMPORTANT: Do NOT recompute H in MATLAB float64.
% The ONN masks were trained against float32 H (torch.fft, float32 precision).
% At k·z ≈ 590,000 rad, float32 phase ULP ≈ 0.07 rad.  Recomputing in float64
% gives different phases → wrong zone wins → ~43% accuracy instead of ~91%.
% Solution: load the exact float32-precision H exported from Python.
fprintf('\nLoading transfer function...\n');
load('matlab_export/H.mat');   % H [200×200 complex double, values from float32]

H_fft = H;  % Already in FFT natural order (Python used fftfreq, no fftshift needed)
k        = 2 * pi / wavelength;   % still needed for display only
fprintf('  Transfer function loaded: %s\n', mat2str(size(H_fft)));
fprintf('  Propagating modes: %d\n', sum(H_fft(:) ~= 0));

%% ========== 3. PROPAGATION FUNCTION ==========
% ASM propagation: FFT → multiply H → IFFT (matches Python physics.py exactly)
propagate = @(field) ifft2(fft2(field) .* H_fft);

%% ========== 4. SIMULATE ONE DIGIT (VISUALIZATION) ==========
fprintf('\n========== SINGLE DIGIT SIMULATION ==========\n');

% Pick digit 0 as example
input_image = double(digit0_sample0);  % [200x200], values in [0,1]

% Step-by-step propagation
field = double(input_image);       % Input amplitude
field_after_mask1 = field .* mask1; % Apply mask 1
field_after_prop1 = propagate(field_after_mask1);  % Propagate 50mm
field_after_mask2 = field_after_prop1 .* mask2;     % Apply mask 2
field_output = propagate(field_after_mask2);         % Propagate 50mm

% Detector intensity
intensity = abs(field_output).^2;

fprintf('  Input energy:    %.4f\n', sum(input_image(:).^2));
fprintf('  Output max:      %.6e\n', max(intensity(:)));
fprintf('  Output total:    %.6e\n', sum(intensity(:)));

% --- Plot propagation stages ---
figure('Name', 'ONN Propagation Stages', 'Position', [100 100 1400 800]);

subplot(2,4,1);
imagesc(input_image); axis image; colorbar;
title('1. Input (Digit 0)'); colormap(gca, 'gray');

subplot(2,4,2);
imagesc(mask1); axis image; colorbar;
title('2. Mask 1'); colormap(gca, 'gray');

subplot(2,4,3);
imagesc(abs(field_after_mask1).^2); axis image; colorbar;
title('3. After Mask 1');

subplot(2,4,4);
imagesc(abs(field_after_prop1).^2); axis image; colorbar;
title('4. After 50mm Propagation');

subplot(2,4,5);
imagesc(mask2); axis image; colorbar;
title('5. Mask 2'); colormap(gca, 'gray');

subplot(2,4,6);
imagesc(abs(field_after_mask2).^2); axis image; colorbar;
title('6. After Mask 2');

subplot(2,4,7);
imagesc(intensity); axis image; colorbar;
title('7. Detector (Intensity)');

subplot(2,4,8);
imagesc(log10(intensity + 1e-12)); axis image; colorbar;
title('8. Detector (Log Scale)');

sgtitle('ONN Propagation — MATLAB ASM Simulation', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, 'matlab_export/propagation_stages.png');
fprintf('  Saved: matlab_export/propagation_stages.png\n');

%% ========== 5. ZONE ANALYSIS ==========
fprintf('\n========== ZONE ANALYSIS ==========\n');

grid_sz = double(grid_size);  % 5
zone_size = N / grid_sz;      % 40 pixels per zone

zone_intensities = zeros(1, grid_sz^2);
for zy = 1:grid_sz
    for zx = 1:grid_sz
        r1 = (zy-1)*zone_size + 1;
        r2 = zy*zone_size;
        c1 = (zx-1)*zone_size + 1;
        c2 = zx*zone_size;
        zone_idx = (zy-1)*grid_sz + zx;  % 1-indexed
        zone_intensities(zone_idx) = sum(sum(intensity(r1:r2, c1:c2)));
    end
end

[~, winner] = max(zone_intensities);
predicted_digit = zone_to_digit(winner);  % 0-indexed digit

fprintf('  Zone intensities:\n');
for z = 1:grid_sz^2
    fprintf('    Zone %2d → digit %d: %.6e%s\n', z, zone_to_digit(z), ...
        zone_intensities(z), char(' *' * (z == winner) + ' ' * (z ~= winner)));
end
fprintf('  Winner: Zone %d → Predicted digit: %d (True: 0)\n', winner, predicted_digit);

% --- Plot zone map ---
figure('Name', 'Zone Intensities');
zone_grid = reshape(zone_intensities, [grid_sz, grid_sz]);
imagesc(zone_grid); axis image; colorbar;
title('Zone Intensity Map (Digit 0)');
for zy = 1:grid_sz
    for zx = 1:grid_sz
        z = (zy-1)*grid_sz + zx;
        text(zx, zy, sprintf('D%d\n%.1e', zone_to_digit(z), zone_intensities(z)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8);
    end
end
saveas(gcf, 'matlab_export/zone_map_digit0.png');

%% ========== 5b. COMPARE WITH PYTHON REFERENCE ==========
fprintf('\n========== PYTHON REFERENCE COMPARISON ==========\n');
if exist('matlab_export/python_reference.mat', 'file')
    ref = load('matlab_export/python_reference.mat');
    
    % Compare input image
    % In Python: img0 [200x200] saved by scipy → MATLAB ref.input_image is [200x200] (transposed from python)
    % Our input_image = digit0_sample0' which should equal python's img0
    % So: ref.input_image (MATLAB, transposed) should equal our input_image.'
    input_match_t  = norm(input_image - ref.input_image', 'fro') / (norm(ref.input_image(:)) + 1e-12);
    input_match_nt = norm(input_image - ref.input_image,  'fro') / (norm(ref.input_image(:)) + 1e-12);
    fprintf('  Input vs ref'' (with transpose):  rel_err=%.4e\n', input_match_t);
    fprintf('  Input vs ref  (no transpose):    rel_err=%.4e\n', input_match_nt);
    if input_match_t < input_match_nt
        fprintf('  --> Input orientation: WITH transpose matches Python\n');
    else
        fprintf('  --> Input orientation: WITHOUT transpose matches Python\n');
    end
    
    % Compare masks
    mask1_match_t  = norm(mask1 - ref.mask1_py', 'fro') / (norm(ref.mask1_py(:)) + 1e-12);
    mask1_match_nt = norm(mask1 - ref.mask1_py,  'fro') / (norm(ref.mask1_py(:)) + 1e-12);
    fprintf('  Mask1 vs ref'' (with transpose):  rel_err=%.4e\n', mask1_match_t);
    fprintf('  Mask1 vs ref  (no transpose):    rel_err=%.4e\n', mask1_match_nt);
    if mask1_match_t < mask1_match_nt
        fprintf('  --> Mask1 orientation: WITH transpose matches Python\n');
    else
        fprintf('  --> Mask1 orientation: WITHOUT transpose matches Python\n');
    end
    
    % Compare output intensity
    intensity_match_t  = norm(intensity - ref.intensity_ref', 'fro') / (norm(ref.intensity_ref(:)) + 1e-12);
    intensity_match_nt = norm(intensity - ref.intensity_ref,  'fro') / (norm(ref.intensity_ref(:)) + 1e-12);
    fprintf('  Intensity vs ref'' (with transpose):  rel_err=%.4e\n', intensity_match_t);
    fprintf('  Intensity vs ref  (no transpose):    rel_err=%.4e\n', intensity_match_nt);
    if intensity_match_t < intensity_match_nt
        fprintf('  --> Intensity orientation: WITH transpose matches Python\n');
    else
        fprintf('  --> Intensity orientation: WITHOUT transpose matches Python\n');
    end
    
    % Compare zone intensities  
    zi_ref = double(ref.zone_ints_ref(:));  % force column [25x1]; python 0-indexed
    fprintf('\n  Zone-by-zone comparison (Python 0-idx vs MATLAB 1-idx):\n');
    for z = 1:25
        mark = '';
        if z-1 == ref.winner_zone; mark = ' <-- Python winner'; end
        if z == winner;            mark = [mark ' <-- MATLAB winner']; end
        fprintf('    Z%2d(py)/Z%2d(ml): py=%.4e  ml=%.4e  ratio=%.3f%s\n', ...
            z-1, z, zi_ref(z), zone_intensities(z), ...
            zone_intensities(z)/(zi_ref(z)+1e-20), mark);
    end
    
    fprintf('\n  Python winner zone (0-indexed): %d → digit %d\n', ...
        ref.winner_zone, zone_to_digit(ref.winner_zone + 1));
    fprintf('  MATLAB winner zone (1-indexed): %d → digit %d\n', ...
        winner, predicted_digit);
else
    fprintf('  python_reference.mat not found — run export_matlab.py first\n');
end

%% ========== 6. ALL 10 DIGITS ==========
fprintf('\n========== ALL 10 DIGITS ==========\n');

figure('Name', 'All Digits — Detector Output', 'Position', [100 100 1200 500]);

digit_names = {'digit0_sample0', 'digit1_sample0', 'digit2_sample0', ...
               'digit3_sample0', 'digit4_sample0', 'digit5_sample0', ...
               'digit6_sample0', 'digit7_sample0', 'digit8_sample0', ...
               'digit9_sample0'};

for d = 0:9
    input_img = double(eval(digit_names{d+1}));  % scipy preserves orientation, no transpose needed
    
    % Forward pass
    f = double(input_img) .* mask1;
    f = propagate(f);
    f = f .* mask2;
    f = propagate(f);
    I_out = abs(f).^2;
    
    % Zone classification
    zi = zeros(1, grid_sz^2);
    for zy = 1:grid_sz
        for zx = 1:grid_sz
            r1 = (zy-1)*zone_size + 1; r2 = zy*zone_size;
            c1 = (zx-1)*zone_size + 1; c2 = zx*zone_size;
            zi((zy-1)*grid_sz + zx) = sum(sum(I_out(r1:r2, c1:c2)));
        end
    end
    [~, w] = max(zi);
    pred = zone_to_digit(w);
    
    subplot(2, 5, d+1);
    imagesc(I_out); axis image; colorbar;
    if pred == d
        title(sprintf('Digit %d → %d ✓', d, pred), 'Color', 'g');
    else
        title(sprintf('Digit %d → %d ✗', d, pred), 'Color', 'r');
    end
    
    fprintf('  Digit %d: predicted %d %s\n', d, pred, ...
        char('✓' * (pred == d) + '✗' * (pred ~= d)));
end

sgtitle('Detector Intensity — All Digits (MATLAB ASM)', 'FontSize', 14);
saveas(gcf, 'matlab_export/all_digits_detector.png');
fprintf('  Saved: matlab_export/all_digits_detector.png\n');

%% ========== 7. FULL TEST SET (10,000 images) ==========
fprintf('\n========== FULL TEST SET VALIDATION ==========\n');
fprintf('Loading 10,000 test images...\n');

load('matlab_export/test_set.mat');

% Print actual loaded shape to determine scipy dimension ordering
fprintf('  x_test size in MATLAB: %s\n', mat2str(size(x_test)));
fprintf('  y_test size in MATLAB: %s\n', mat2str(size(y_test)));

% Robustly find which dimension is 10000 (test set size)
xsz = size(x_test);
if xsz(1) == 10000
    num_test = xsz(1);
    get_img = @(n) squeeze(double(x_test(n, :, :)));  % no transpose needed
elseif xsz(3) == 10000
    num_test = xsz(3);
    get_img = @(n) double(x_test(:, :, n));  % no transpose needed
else
    error('Unexpected x_test shape: %s', mat2str(xsz));
end

correct = 0;
predictions = zeros(num_test, 1);

fprintf('Running inference...\n');
tic;
for n = 1:num_test
    img = get_img(n);  % [200x200], correct Python orientation
    
    % Forward pass
    f = double(img) .* mask1;
    f = propagate(f);
    f = f .* mask2;
    f = propagate(f);
    I_out = abs(f).^2;
    
    % Zone classification
    zi = zeros(1, grid_sz^2);
    for zy = 1:grid_sz
        for zx = 1:grid_sz
            r1 = (zy-1)*zone_size + 1; r2 = zy*zone_size;
            c1 = (zx-1)*zone_size + 1; c2 = zx*zone_size;
            zi((zy-1)*grid_sz + zx) = sum(sum(I_out(r1:r2, c1:c2)));
        end
    end
    [~, w] = max(zi);
    pred = zone_to_digit(w);
    predictions(n) = pred;
    
    if pred == y_test(n)  % y_test is (10000,1) or (1,10000), linear indexing works for both
        correct = correct + 1;
    end
    
    if mod(n, 1000) == 0
        elapsed = toc;
        fprintf('  %d/%d done (%.1f sec, current acc: %.2f%%)\n', ...
            n, num_test, elapsed, 100*correct/n);
    end
end

elapsed = toc;
accuracy = 100 * correct / num_test;
fprintf('\n===================================\n');
fprintf('MATLAB ASM Accuracy: %.2f%% (%d/%d)\n', accuracy, correct, num_test);
fprintf('Time: %.1f seconds (%.1f ms/image)\n', elapsed, 1000*elapsed/num_test);
fprintf('===================================\n');

%% ========== 8. CONFUSION MATRIX ==========
fprintf('\nGenerating confusion matrix...\n');

cm = confusionmat(double(y_test), predictions);
figure('Name', 'Confusion Matrix');
confusionchart(cm, 0:9);
title(sprintf('MATLAB ASM Validation — Accuracy: %.2f%%', accuracy));
saveas(gcf, 'matlab_export/confusion_matrix_matlab.png');
fprintf('  Saved: matlab_export/confusion_matrix_matlab.png\n');

% Per-class accuracy
fprintf('\nPer-class accuracy:\n');
for d = 0:9
    mask_d = (y_test == d);
    n_d = sum(mask_d);
    correct_d = sum(predictions(mask_d) == d);
    fprintf('  Digit %d: %d/%d = %.1f%%\n', d, correct_d, n_d, 100*correct_d/n_d);
end

fprintf('\nDone! All results saved to matlab_export/\n');
