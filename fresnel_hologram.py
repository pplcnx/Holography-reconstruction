import numpy as np
import matplotlib.pyplot as plt


def create_test_pattern(size=512):
    """Generate a simple circular aperture pattern."""
    x = np.linspace(-1.0, 1.0, size)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    pattern = (R < 0.5).astype(float)
    return pattern


def angular_spectrum(field, wavelength, distance, pixel_size):
    """Propagate a field using the angular spectrum method."""
    k = 2 * np.pi / wavelength
    N, M = field.shape
    fx = np.fft.fftfreq(M, d=pixel_size)
    fy = np.fft.fftfreq(N, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(1j * distance * np.sqrt(np.maximum(0, k**2 - (2 * np.pi * FX)**2 - (2 * np.pi * FY)**2)))
    F = np.fft.fft2(field)
    return np.fft.ifft2(F * H)


def main():
    wavelength = 633e-9  # red light (633 nm)
    pixel_size = 8e-6    # 8 micrometers
    distance = 0.2       # propagation distance in meters

    # Desired amplitude pattern at the target plane
    target_amp = create_test_pattern(512)
    target_field = target_amp * np.exp(1j * 0)

    # Back propagate from target to SLM plane
    field_slm = angular_spectrum(target_field, wavelength, -distance, pixel_size)

    # Phase-only hologram to load on SLM
    hologram_phase = np.angle(field_slm)
    hologram = np.exp(1j * hologram_phase)

    # Forward propagate from SLM to target plane
    reconstructed = angular_spectrum(hologram, wavelength, distance, pixel_size)

    # Visualization
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(target_amp, cmap="gray")
    axs[0].set_title("Target amplitude")
    axs[0].axis('off')

    axs[1].imshow(hologram_phase, cmap="gray")
    axs[1].set_title("Hologram phase")
    axs[1].axis('off')

    axs[2].imshow(np.abs(reconstructed), cmap="gray")
    axs[2].set_title("Reconstruction amplitude")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
