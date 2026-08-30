# E1 Branching Calculator

A self-contained Python calculator for angular electric-dipole (E1) line
strengths and branching ratios between one-electron, spin-orbit-resolved
atomic subshells.

It evaluates the required Wigner 3-j and 6-j symbols directly with Racah
factorial formulas. There are no third-party dependencies: only the Python
standard library is used.

The emission branching ratio intensities can be verified with data tabulated in Hephaestus.

## Scope

The state model is

```math
|n\,l\,s\,j\,m_j\rangle, \qquad s=\tfrac12.
```

Here, `j` is the total angular momentum of one electron, not its spin
projection. This is not a general many-electron atomic-term calculator.

The default quantity is the complete-multiplet angular line strength:

```math
S_{if}^{(\mathrm{ang})} =
(2j_i+1)(2j_f+1)
\begin{Bmatrix}
l_f & j_f & \tfrac12\\
j_i & l_i & 1
\end{Bmatrix}^{2}
|\langle l_f\|C^{(1)}\|l_i\rangle|^2.
```

It is summed over all initial and final magnetic substates and photon
polarizations, with the radial integral factored out. It is therefore an
angular strength, not by itself an absolute transition rate or measured
intensity.

## Requirements

- Python 3.10 or newer
- No NumPy, SciPy, SymPy, or internet connection

## Quick start

Clone the repository and enter it:

```bash
git clone https://github.com/KlausMeng/E1-branching-calculator.git
cd E1-branching-calculator
```

Run the built-in tests:

```bash
python e1_branching.py --self-test
```

## Shell labels and direction

The calculator accepts labels including `K`, `L1`-`L3`, `M1`-`M5`,
`N1`-`N7`, and the corresponding O, P, and Q labels. Explicit dictionaries
can also be used through the Python API.

Tuple and CLI order always means the electron direction:

```text
from_state -> to_state
```

For example, absorption from `2p3/2` into `3d5/2` is `L3 -> M5`.
Fluorescence notation commonly writes the core hole first: the line called
`L3-M5` is emission in the reverse electron direction, `M5 -> L3`.

## CLI examples

### One angular channel

```bash
python e1_branching.py channel L3 M5
```

This reports

```text
S_ang = 2.4
```

The reverse emission channel has the same complete-multiplet sum:

```bash
python e1_branching.py channel M5 L3
```

### Angular branching ratio

```bash
python e1_branching.py ratio L3:M5 L3:M4
```

The result is

```text
L3 -> M5 : L3 -> M4 = 9 : 1
```

This is an intensity ratio. The corresponding magnitude-amplitude ratio is
`3:1`. The `9:1` result arises from angular-momentum recoupling, not from the
simple final degeneracy ratio `6:4`.

For emission, reverse the electron directions:

```bash
python e1_branching.py ratio M5:L3 M4:L3
```

### Statistical L2:L3 absorption ratio

```bash
python e1_branching.py edge-ratio \
  --initial L2 L3 \
  --final M4 M5
```

With filled initial subshells and completely vacant specified final
subshells, the calculation derives

```text
L2 -> M4 = 5 relative units
L2 -> M5 = 0 (E1 forbidden)
L3 -> M4 = 1 relative unit
L3 -> M5 = 9 relative units

L2 total : L3 total = 5 : 10 = 1 : 2
```

### Explicit electron and hole counts

Absorption weighting is

```math
S_{if}^{(\mathrm{weighted})} =
S_{if}^{(\mathrm{ang})}
\frac{N_i}{2j_i+1}
\frac{H_f}{2j_f+1}.
```

Counts can be supplied explicitly:

```bash
python e1_branching.py edge-ratio \
  --initial L2 L3 \
  --final O4 O5 \
  --initial-electrons L2=2 \
  --initial-electrons L3=4 \
  --final-holes O4=2 \
  --final-holes O5=3
```

This example treats five `5d` holes as uniformly distributed between the
`5d3/2` (`O4`) and `5d5/2` (`O5`) subshells. Real spin-orbit-resolved hole
counts should be used when known.

Fully occupied target subshells have zero holes and are Pauli blocked. For
example:

```bash
python e1_branching.py edge-ratio \
  --initial L2 L3 \
  --final M4 M5 \
  --final-holes M4=0 \
  --final-holes M5=0
```

### Explicit photon-energy conventions

Angular-only calculations apply no photon-energy factor. An
oscillator-strength-like absorption comparison may apply a factor
proportional to `omega`:

```bash
python e1_branching.py channel L3 M5 \
  --photon-energy 700 \
  --energy-weighting absorption
```

A spontaneous E1 emission rate carries a factor proportional to `omega^3`:

```bash
python e1_branching.py channel M5 L3 \
  --photon-energy 700 \
  --energy-weighting emission
```

Photon energies must use one consistent unit within a comparison. The
reported corrected strengths remain proportional quantities unless the
radial matrix elements and physical constants are also supplied.

## Python API

```python
from e1_branching import (
    CorrectionFactors,
    Transition,
    absorption_edge_ratios,
    angular_line_strength,
    branching_ratios,
    e1_allowed,
    parse_subshell,
)

state = parse_subshell({"n": 2, "l": "p", "j": "3/2"})
allowed, reason = e1_allowed("L3", "M5")

line = angular_line_strength(("L3", "M5"))
print(line.wigner_6j, line.orbital_3j, line.angular_strength)

branches = branching_ratios([
    ("L3", "M5"),
    ("L3", "M4"),
])
print(branches.relative_ratio)  # (9, 1)

edges = absorption_edge_ratios(
    initial_edges=["L2", "L3"],
    final_states=["O4", "O5"],
    initial_electrons={"L2": 2, "L3": 4},
    final_holes={"O4": 2, "O5": 3},
)
```

Per-transition radial, photon-energy, and extra multiplicative factors are
explicit:

```python
emission = Transition(
    "M5",
    "L3",
    CorrectionFactors(
        radial_matrix_element=1.0,
        photon_energy=8652.0,
        energy_weighting="emission",
        weight=1.0,
    ),
)

result = angular_line_strength(emission)
print(result.angular_strength)
print(result.corrected_strength)
```

## What each normalization means

| Quantity | Meaning | Includes occupations? | Includes radial/energy factors? |
|---|---|---:|---:|
| `angular_strength` | Complete magnetic-substate and polarization sum | No | No |
| Weighted edge strength | Uniformly occupied initial states and uniformly distributed final holes | Yes | Only if supplied |
| Absolute Einstein A coefficient | Physical spontaneous-emission rate | Yes | Requires radial factor, `omega^3`, and constants |
| Radiative branching fraction | One radiative channel divided by all competing radiative channels | Yes | Yes |
| Measured photon counts | Experimental spectrum | Yes | Also needs hole production, fluorescence yield, attenuation, and detector response |

When comparing with fluorescence databases such as Hephaestus, remember that
line strengths are commonly normalized separately within each core-hole edge.
Two values belonging to different edges do not generally share the same
denominator.

## Simplifications and physical limitations

The calculator deliberately uses a transparent independent-particle model:

- Every state is a one-electron `n l j` subshell with fixed `s=1/2`.
- The E1 operator is spin independent and enforces `Delta l = +/-1`,
  `Delta j = 0, +/-1`, excluding `j=0 <-> j=0`.
- The default is isotropic/unpolarized and sums complete magnetic multiplets.
- Partial electrons and holes are assumed uniform across the corresponding
  `m_j` substates.
- Radial integrals are removed by default. They cancel approximately only
  when genuinely common to the compared channels.
- Photon-energy differences are ignored unless an explicit `omega` or
  `omega^3` convention is selected.
- Even spin-orbit partners can have slightly different relativistic radial
  functions, especially for heavy elements.
- The model does not include many-electron multiplets, configuration
  interaction, crystal fields, band structure, covalency, polarization,
  nonuniform hole populations, Coster-Kronig transfer, Auger competition,
  fluorescence yields, self-absorption, or detector response.

Consequently, the results are useful angular limits and consistency checks.
They are not substitutes for element-specific relativistic atomic calculations
or a many-body description of partially occupied `d` and `f` shells.

## Verification

```bash
python e1_branching.py --self-test
```

The built-in suite checks label parsing, E1 selection rules, reversal
symmetry, invalid populations, the `9:1` and `5:1:9` angular ratios, and the
statistical `L2:L3 = 1:2` absorption limit.
