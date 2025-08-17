# pip install pynucastro numpy
import numpy as np
import pynucastro as pyna

# --- inputs: plasma state (example tokamak-ish numbers) ---
T_keV = 15.0                     # core temperature (keV)
T_GK  = T_keV * 1.16045e7 / 1e9  # convert keV -> GK
ne = 1.0e20                      # electron density [m^-3]
np_ = 0.5e20                     # proton density  [m^-3]
n3He = 1.0e10                    # trace 3He [m^-3] (set from your mix)
NA = 6.02214076e23

# --- load ReacLib and pick reactions ---
rl = pyna.ReacLibLibrary()

# helpers to fetch specific rates by reagents/products labels:
def get_rate(label):
    # label examples: "p + p -> d + e+ + nu_e" (approx), "he3 + p -> he4 + e+ + nu_e" (hep)
    matches = [r for r in rl.get_rates() if label.lower() in r.fname().lower()]
    if not matches:
        raise ValueError(f"Rate not found for: {label}")
    return matches[0]

# Try to find pp and hep (naming varies slightly across ReacLib versions)
pp_candidates  = ["p + p -> d + e+ + nu_e", "h1 + h1 -> he2 + ..."]  # relaxed search
hep_candidates = ["he3 + p -> he4 + e+ + nu_e"]

def find_first(cands):
    for s in cands:
        try: return get_rate(s)
        except: pass
    raise RuntimeError(f"Could not locate any of: {cands}")

pp  = find_first(pp_candidates)
hep = find_first(hep_candidates)

# ReacLib returns NA<σv>(T) in cm^3/mol/s. Convert to <σv> in m^3/s:
def NA_sv_to_m3_s(NA_sv_cm3_mol_s):
    return (NA_sv_cm3_mol_s / NA) * 1.0e-6  # cm^3→m^3 and divide by Avogadro

pp_NAsv  = pp.eval(T_GK)      # cm^3/mol/s
hep_NAsv = hep.eval(T_GK)

pp_sv_m3s  = NA_sv_to_m3_s(pp_NAsv)
hep_sv_m3s = NA_sv_to_m3_s(hep_NAsv)

# Rate densities [m^-3 s^-1]
R_pp  = 0.5 * np_**2 * pp_sv_m3s            # factor 1/2 for identical reactants
R_hep = n3He * np_ * hep_sv_m3s

# --- pep via Bahcall relation (fallback) ---
# R_pep ≈ C(T)*ne*np_^2, or equivalently R_pep ≈ (pep/pp)*R_pp with weak T,ρ dependence.
# Use solar-core ratio as a baseline and scale ~ ne / sqrt(T). (See Bahcall 1969.)
pep_over_pp_solar = 2.3e-3
scale = (ne / 6e31) * np.sqrt(1.55 / T_GK)   # normalize to solar core (ne~6e31 m^-3, T~1.55 GK)
R_pep = R_pp * pep_over_pp_solar * scale

print(f"T = {T_keV:.1f} keV")
print(f"R_pp  ~ {R_pp:.3e} m^-3 s^-1")
print(f"R_pep ~ {R_pep:.3e} m^-3 s^-1 (scaled from Bahcall)")
print(f"R_hep ~ {R_hep:.3e} m^-3 s^-1")
