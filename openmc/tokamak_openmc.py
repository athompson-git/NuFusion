"""
OpenMC fixed-source example: DT tokamak surrogate for neutron flux / activation inputs

What this does
--------------
- Builds a simple concentric-cylinder surrogate of a DT tokamak radial stack:
  Plasma (void) -> First Wall (SS316) -> Breeder/Blanket (LiPb + Be) -> Shield (Water+Steel mix) -> Concrete bioshield
- Places a monoenergetic isotropic 14.1 MeV DT neutron source at the plasma center (point source).
- Tallies energy-dependent neutron flux in each region for activation post-processing.
- Optionally tallies a few reaction rates (n,gamma), (n,p), (n,alpha) per cell (total, not per-nuclide),
  mainly as a sanity check.
- Dumps flux spectra to CSV for downstream activation codes (e.g. OpenMC-Activation / ALARA).


Outputs
-------
- statepoint.N.h5              # OpenMC tally results
- flux_cell_spectra.csv        # groupwise neutron flux (per cell)
- region_metadata.csv          # cell -> material, volume, region label
- quick_reaction_rates.csv     # integrated (n,gamma)/(n,p)/(n,alpha) per cell (sanity check)
"""

import numpy as np
import pandas as pd
import openmc

########################
# Geometry & materials #
########################

def build_materials():
    mats = openmc.Materials()

    # Stainless steel 316L (approximate weight fractions)
    ss316 = openmc.Material(name="SS316L", temperature=600)
    ss316.set_density("g/cm3", 8.0)
    ss316.add_element('Fe', 0.68)
    ss316.add_element('Cr', 0.18)
    ss316.add_element('Ni', 0.12)
    ss316.add_element('Mo', 0.02)
    ss316.add_element('Mn', 0.01)
    ss316.add_element('C',  0.001)

    # Beryllium multiplier (Be metal)
    be = openmc.Material(name="Be", temperature=600)
    be.set_density("g/cm3", 1.85)
    be.add_element('Be', 1.0)

    # LiPb eutectic breeder (Li17-Pb83), assume Li-6 enrichment for T-breeding (e.g., 50% Li-6)
    lipb = openmc.Material(name="Li17Pb83", temperature=800)
    lipb.set_density("g/cm3", 10.2)
    # Atomic fractions: 17% Li, 83% Pb
    # Split Li between Li-6 and Li-7
    lipb.add_nuclide('Li6', 0.17 * 0.50)
    lipb.add_nuclide('Li7', 0.17 * 0.50)
    lipb.add_element('Pb', 0.83)

    # Water (shield coolant)
    water = openmc.Material(name="H2O", temperature=350)
    water.set_density("g/cm3", 0.997)
    water.add_element('H', 2.0)
    water.add_element('O', 1.0)

    # Concrete bioshield (very rough)
    concrete = openmc.Material(name="Concrete", temperature=300)
    concrete.set_density("g/cm3", 2.3)
    concrete.add_element('O', 0.52)
    concrete.add_element('Si', 0.325)
    concrete.add_element('Ca', 0.06)
    concrete.add_element('Al', 0.045)
    concrete.add_element('Fe', 0.02)

    # Water+Steel mix (shield) — volume fractions: 60% water, 40% steel (simple mixture)
    shield_mix = openmc.Material.mix_materials(
        [water, ss316], fracs=[0.60, 0.40], percent_type='vo', name='ShieldMix')

    mats.extend([ss316, be, lipb, water, concrete, shield_mix])
    return mats


def build_geometry(mats):
    # Radii (m)
    r_plasma   = 0.50
    r_fw       = r_plasma + 0.02   # first wall thickness 2 cm
    r_be       = r_fw     + 0.05   # Be multiplier 5 cm
    r_blanket  = r_be     + 0.80   # breeder blanket 80 cm
    r_shield   = r_blanket+ 0.80   # shield 80 cm
    r_bio      = r_shield + 1.50   # concrete bioshield 1.5 m

    height = 2.0  # m (axial height)

    # Surfaces
    cyl_plasma  = openmc.ZCylinder(r=r_plasma)
    cyl_fw      = openmc.ZCylinder(r=r_fw)
    cyl_be      = openmc.ZCylinder(r=r_be)
    cyl_blanket = openmc.ZCylinder(r=r_blanket)
    cyl_shield  = openmc.ZCylinder(r=r_shield)
    cyl_bio     = openmc.ZCylinder(r=r_bio)

    zmin = openmc.ZPlane(z0=-height/2)
    zmax = openmc.ZPlane(z0= height/2)

    # Vacuum boundary
    outer_boundary = openmc.Sphere(r=r_bio*2.0, boundary_type='vacuum')

    # Regions (finite height cylinder stack)
    region_plasma  = -cyl_plasma  & +zmin & -zmax
    region_fw      = +cyl_plasma  & -cyl_be   & +zmin & -zmax  # includes Be; we'll split below
    region_be_only = +cyl_fw      & -cyl_be   & +zmin & -zmax
    region_fw_only = +cyl_plasma  & -cyl_fw   & +zmin & -zmax

    region_blanket = +cyl_be      & -cyl_blanket & +zmin & -zmax
    region_shield  = +cyl_blanket & -cyl_shield  & +zmin & -zmax
    region_bio     = +cyl_shield  & -cyl_bio     & +zmin & -zmax
    region_outside = +cyl_bio     & -outer_boundary

    # Cells
    c_plasma   = openmc.Cell(region=region_plasma)
    c_fw       = openmc.Cell(region=region_fw_only,   fill=mats["SS316L"])  # first wall steel
    c_be       = openmc.Cell(region=region_be_only,   fill=mats["Be"])      # Be multiplier
    c_blanket  = openmc.Cell(region=region_blanket,   fill=mats["Li17Pb83"])# breeder
    c_shield   = openmc.Cell(region=region_shield,    fill=mats["ShieldMix"])# shield
    c_bio      = openmc.Cell(region=region_bio,       fill=mats["Concrete"]) # bioshield
    c_out      = openmc.Cell(region=region_outside)   # vacuum to boundary

    univ = openmc.Universe(cells=[c_plasma, c_fw, c_be, c_blanket, c_shield, c_bio, c_out])

    geom = openmc.Geometry(univ)
    return geom, {
        'plasma': c_plasma,
        'first_wall': c_fw,
        'be': c_be,
        'blanket': c_blanket,
        'shield': c_shield,
        'bioshield': c_bio,
    }

############################
# Source & simulation setup #
############################

def build_settings():
    settings = openmc.Settings()
    settings.run_mode = 'fixed source'

    # 14.1 MeV DT point source at the center (isotropic)
    src = openmc.Source()
    src.space = openmc.stats.Point((0.0, 0.0, 0.0))
    src.angle = openmc.stats.Isotropic()
    src.energy = openmc.stats.Discrete([14.1e6], [1.0])
    settings.source = src

    settings.particles = int(3e6)   # adjust as needed
    settings.batches = 30
    settings.output = {'tallies': True}
    settings.max_lost_particles = int(1e7)
    return settings

###########
# Tallies #
###########

def build_tallies(cells):
    tallies = openmc.Tallies()

    # Energy bins (eV) — 0.01 eV to 20 MeV, logarithmic
    e_min, e_max = 1.0e-2, 20.0e6
    n_bins = 300
    edges = np.logspace(np.log10(e_min), np.log10(e_max), n_bins+1)
    efilt = openmc.EnergyFilter(edges)

    # One tally per cell for flux spectrum
    cell_filters = {name: openmc.CellFilter(cell) for name, cell in cells.items()}

    for name, cf in cell_filters.items():
        t = openmc.Tally(name=f"flux_{name}")
        t.filters = [cf, efilt]
        t.scores = ['flux']
        tallies.append(t)

    # Integrated reaction-rate sanity checks (no energy filter)
    rr = openmc.Tally(name="reaction_rates")
    rr.filters = [openmc.CellFilter(list(cells.values()))]
    # Common activation-related reactions (total over material):
    # OpenMC accepts ENDF MT names; use standard aliases where available
    rr.scores = ['(n,gamma)', '(n,p)', '(n,alpha)']
    tallies.append(rr)

    return tallies, edges

#########################
# Post-processing utils #
#########################

def dump_flux_spectra(sp, edges, cells):
    rows = []
    for name, cell in cells.items():
        tally = sp.get_tally(name=f"flux_{name}")
        # Mean values; normalized per source particle. Convert to per-cm^2 per-source by default.
        vals = tally.mean.ravel()
        # Bin midpoints for convenience
        e_lo = edges[:-1]
        e_hi = edges[1:]
        e_mid = np.sqrt(e_lo * e_hi)
        for el, eh, em, v in zip(e_lo, e_hi, e_mid, vals):
            rows.append({
                'region': name,
                'E_low_eV': el,
                'E_high_eV': eh,
                'E_mid_eV': em,
                'flux_per_source_per_eV': v  # differential flux in bin per source particle
            })
    df = pd.DataFrame(rows)
    df.to_csv('flux_cell_spectra.csv', index=False)
    print("[write] flux_cell_spectra.csv")


def dump_region_metadata(geom, cells):
    rows = []
    for name, cell in cells.items():
        vol = cell.region.bounding_box.volume  # geometric bbox volume (approx upper bound)
        mat = cell.fill.name if isinstance(cell.fill, openmc.Material) else 'void'
        rows.append({'region': name, 'material': mat, 'bbox_volume_m3': vol})
    pd.DataFrame(rows).to_csv('region_metadata.csv', index=False)
    print("[write] region_metadata.csv")


def dump_reaction_rates(sp, cells):
    t = sp.get_tally(name="reaction_rates")
    # Map cell IDs back to names
    id_to_name = {cell.id: name for name, cell in cells.items()}
    rows = []
    for i, c_id in enumerate(t.filters[0].bins):
        name = id_to_name.get(c_id, f"cell_{c_id}")
        vals = t.mean[i, :]
        rows.append({
            'region': name,
            'n_gamma_rate_per_source': float(vals[0]),
            'n_p_rate_per_source': float(vals[1]),
            'n_alpha_rate_per_source': float(vals[2]),
        })
    pd.DataFrame(rows).to_csv('quick_reaction_rates.csv', index=False)
    print("[write] quick_reaction_rates.csv")


########
# Main #
########
if __name__ == "__main__":
    materials = build_materials()
    geometry, cells = build_geometry(materials)
    settings = build_settings()
    tallies, edges = build_tallies(cells)

    materials.export_to_xml()
    geometry.export_to_xml()
    settings.export_to_xml()
    tallies.export_to_xml()

    print("[run] openmc")
    openmc.run()

    # Load results
    sp = openmc.StatePoint("statepoint.{}.h5".format(settings.batches))
    dump_flux_spectra(sp, edges, cells)
    dump_region_metadata(geometry, cells)
    dump_reaction_rates(sp, cells)

    print("Done. Use flux_cell_spectra.csv as activation input (groupwise flux per region).")
