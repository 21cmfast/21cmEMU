# Changelog

## Unreleased: rh / norh (radio heating)

- Two 6-parameter emulators (`rh`, `norh`) of a radio background from mini-halo
  galaxies plus dark matter–baryon scattering, with and without radio
  (soft-photon) heating of the gas.
- Emulate the global neutral fraction, brightness temperature and radio
  temperature at 110 redshifts (4.9 < z < 49), and the Thomson optical depth.
- New `RHEmulatorInput` / `NoRHEmulatorInput`, `RHEmulatorOutput` /
  `NoRHEmulatorOutput`, `RHEmulatorErrors` / `NoRHEmulatorErrors` and
  `RadioHeatingEmulatorProperties`; `Emulator(..., weights_path=...)`.
- The trained weights are distributed separately and are not in the repository.

## v3 (mcg)

- Full 11-parameter emulator for molecular cooling galaxies (Pop II + Pop III).
- Emulates 2D cylindrical power spectrum P(k⊥, k∥) via a score-based diffusion model.
- Emulates global brightness temperature, neutral fraction, spin temperature, optical depth, and UV luminosity functions.

## v2 (radio)

- 5-parameter radio background emulator.
- Emulates radio temperature Tr in addition to the standard global quantities and 1D power spectrum.

## v1 (acg)

- Original 9-parameter emulator for atomic cooling galaxies (Pop II only).
- Emulates 21-cm power spectrum, global brightness temperature, IGM spin temperature, neutral fraction, Thomson optical depth τe, and UV luminosity functions.
