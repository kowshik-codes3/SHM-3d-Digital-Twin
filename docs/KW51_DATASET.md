# KW51 Benchmark Dataset Documentation

The KW51 dataset contains continuous structural health monitoring measurements of the KW51 steel bowstring railway bridge over the Leuven-Mechelen canal in Belgium.

- **Authors**: Kristof Maes & Geert Lombaert (KU Leuven)
- **DOI**: `10.5281/zenodo.3745914`
- **Bridge Type**: Single-track tied-arch (bowstring) railway bridge, span 115 m.

## Archive Contents in `data/kw51/`

1. **`ambient_201810.zip` (1.41 GB)**:
   - 144 `.mat` files of raw ambient vibration measurements recorded in October 2018.
   - Sampling rate: $f_s \approx 825.8\text{ Hz}$.
   - 12 acceleration channels:
     - Deck accelerometers: `aBD11Az`, `aBD17Ay`, `aBD17Az`, `aBD17Cz`, `aBD23Ay`, `aBD23Az`
     - Arch accelerometers: `aAR0910Ay`, `aAR0910Cy`, `aAR1516Ay`, `aAR1516Cy`, `aAR2122Ay`, `aAR2122Cy`
   - Total sample count: ~35.67 million points (~12 hours total ambient record).

2. **`trackedmodes.zip` (12.9 MB)**:
   - `trackedmodes.mat`: 11,328 hourly observations over 15 months.
   - Identified natural frequencies (`f`, 14 modes).
   - Mode shapes (`m`) and modal damping ratios (`xi`).
   - Environmental measurements (`env`): deck temperature (`tBD31A`), ambient air temperature (`tVL`), humidity (`rhBD31A`), and wind velocity.
