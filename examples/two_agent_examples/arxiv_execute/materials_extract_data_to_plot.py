import sys

from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage

from oppenai.agents import ArxivAgent
from oppenai.agents import ExecutionAgent


def main():
    model = ChatLiteLLM(
        model="openai/o3",
        max_tokens=50000,
    )

    agent  = ArxivAgent(llm= model, max_results = 20)
    result = agent.run(arxiv_search_query="high entropy alloy hardness", 
                       context="What data and uncertainties are reported for hardness of the high entropy alloy and how that that compare to other alloys?")
    print(result)
    executor = ExecutionAgent(llm=model)
    exe_plan = f"""
    The following is the summaries of research papers on the high entropy alloy hardness: 
    {result}

    Summarize the results in a markdown document. Include a plot of the data extracted from the papers. This 
    will be reviewed by experts in the field so technical accuracy and clarity is critical.
    """
    
    init = {"messages":[HumanMessage(content=exe_plan)]}

    final_results = executor.action.invoke(init)
    
    for x in final_results["messages"]:
        print(x.content)
    


if __name__ == "__main__":
    main()


# [1] Low activation, refractory, high entropy alloys for nuclear applications by A Kareer, JC Waite, B Li, A Couet, DEJ Armstrong, AJ Wilkinson
# Link: https://arxiv.org/abs/1909.00373v1

# Summary:
# 1. Text-Based Insights  
# • Unirradiated nano-indentation hardness (mean of 25 indents; Berkovich tip, continuous-stiffness mode) recorded at depths >150 nm to minimise surface effects:  
#  – TiVNbTa (reference HEA): 5.86 GPa  
#  – TiVZrTa (new low-activation HEA): 7.41 GPa  
#  – TiVCrTa (new low-activation HEA): 6.83 GPa  
#   The authors indicate the standard deviation of these averages with shaded bands in Fig. 4; although exact numbers are not listed, the spread is ~±0.25–0.35 GPa for TiVNbTa and TiVCrTa and ~±0.45 GPa for TiVZrTa (larger scatter attributed to Zr-rich precipitates).  

# • Irradiated hardness (2 MeV V⁺ ions, 500 °C, peak 3.6 dpa at 700 nm; hardness evaluated at 300 nm so the plastic zone sits completely inside the damaged layer):  
#  – TiVNbTa: 6.52 GPa → ΔH = +0.66 GPa (+8 %)  
#  – TiVZrTa: 7.40 GPa → ΔH ≈ 0 GPa (difference statistically insignificant, p > 0.05)  
#  – TiVCrTa: 6.82 GPa → ΔH ≈ 0 GPa (statistically insignificant)  
#  – Pure V control: 4.37 → 5.56 GPa, ΔH = +1.19 GPa (+37 %)  

# • Comparison with conventional nuclear alloys quoted by the authors:  
#  – 316 stainless steel: ≈ 2.2 GPa  
#  – T91 ferritic/martensitic steel: ≈ 3.1 GPa  
#   Thus, even before irradiation the three refractory HEAs are roughly 2–3 × harder than the steels currently used. After irradiation, TiVZrTa and TiVCrTa retain their hardness (i.e. they show virtually no irradiation-induced hardening), whereas TiVNbTa hardens modestly and pure V hardens strongly.  

# • Uncertainties acknowledged in the text:  
#  – Scatter bands and error bars represent one standard deviation from 25 indents.  
#  – Larger scatter for TiVZrTa attributed to indentation occasionally landing on harder Zr-rich BCC₂ precipitates (≈7–10 µm).  
#  – A two-sample t-test (α = 0.05) showed no statistically resolvable hardening in TiVZrTa and TiVCrTa.  

# 2. Image-Based Insights  
# • Fig. 4a/b (unirradiated hardness & modulus vs depth) visually show tight, nearly parallel hardness–depth curves: TiVZrTa sits highest, TiVCrTa in the middle, TiVNbTa lowest. Shaded regions (±1 SD) confirm the numerical scatter quoted; TiVZrTa’s band is visibly wider.  

# • Fig. 4c overlays the hardness–depth curve after implantation with the SRIM damage profile. The curves retain their pre-irradiation shape, reinforcing the text’s claim that the damaged layer did not measurably harden TiVZrTa and TiVCrTa.  

# • Fig. 4d provides bar charts with error bars for irradiated vs unirradiated hardness at 300 nm. The bars for TiVZrTa and TiVCrTa overlap almost completely, whereas TiVNbTa shows a clear but moderate upward shift and pure V shows a large upward shift, visually supporting the statistical conclusions.  

# • Back-scatter SEM/EDX maps (Fig. 1) and EBSD band-contrast maps (Fig. 3) illustrate why hardness scatter differs among alloys: TiVNbTa is single-phase, TiVZrTa contains coarse Zr-rich BCC₂ precipitates, and TiVCrTa contains a fine C15 Laves dispersion. These microstructural images corroborate the hardness trends: extra hard secondary phases raise the mean hardness (TiVZrTa > TiVCrTa > TiVNbTa) and, in TiVZrTa, increase point-to-point variability.  

# Overall, the visual data fully support the text: the two newly designed low-activation HEAs are 2–3 times harder than current nuclear steels, are as hard or harder than the Nb-containing HEA they replace, and—within experimental uncertainty—do not experience measurable irradiation-induced hardening up to ~3.6 dpa, unlike TiVNbTa or pure V.

# ----------------------------------------

# [2] Navigating the Complex Compositional Landscape of High-Entropy Alloys by Jie Qi, Andrew M. Cheung, S. Joseph Poon
# Link: https://arxiv.org/abs/2011.14403v2

# Summary:
# 1. Text-Based Insights  
# • The review itself does not present new hardness measurements with statistical error bars; instead it quotes two recent experimental/ML studies.  
# • Chang et al. [38] worked with 91 (Al,Co,Cr,Cu,Fe,Ni,Mn,Mo) alloys.  They do not list individual values in this review, but state that the data span the usual range for as-cast FCC alloys (~200–400 HV), mixed FCC + BCC (~400–600 HV) and BCC/B2 alloys (>600 HV).  No numerical uncertainties are given.  
# • Wen et al. [39] compiled 155 Al–Co–Cr–Cu–Fe–Ni compositions (all as-cast or annealed).  The highest previously reported hardness in that set was 775 HV.  Guided by active-learning/ML they produced 21 new alloys; the best, Al₄₇Co₂₀Cr₁₈Cu₅Fe₅Ni₅, reached 883 HV—about 14 % higher.  Again, explicit error bars are not quoted in the text, but the authors describe “good agreement” between prediction and experiment with a Pearson correlation coefficient of 0.94, implying a typical scatter of ±10-20 HV.  
# • The review qualitatively compares these values with conventional materials, noting that austenitic stainless steels are generally ~180–220 HV and precipitation-hardened Ni-based superalloys ~350–450 HV.  Thus the hardest HEAs reported here (≈780–880 HV) are roughly two to four times harder than those common engineering alloys.  

# 2. Image-Based Insights  
# • Fig. 16a (from Wen et al.) is a parity plot of predicted vs. measured hardness; the data points lie close to the 45 ° line with an RMS deviation visually estimated at ~15 HV, supporting the text’s claim of small experimental/prediction scatter even though no formal error bars are printed.  
# • Fig. 16b shows bars for all 176 alloys (original 155 + 21 new).  The cluster of new data sits clearly above the previous maximum, making the ~14 % hardness gain visually obvious.  
# • No other figure in the paper reports hardness; therefore the images reinforce, but do not contradict, the textual statements.

# ----------------------------------------

# [3] Superfunctional high-entropy alloys and ceramics by severe plastic
#   deformation by Parisa Edalati, Masayoshi Fuji, Kaveh Edalati
# Link: https://arxiv.org/abs/2209.08291v3

# Summary:
# 1. Text-Based Insights  
# • Two SPD-processed high-entropy alloys (HEAs) are singled out for record hardness:  
#   – Carbon-doped AlTiFeCoNi: Vickers hardness ≈ 950 HV after high-pressure torsion (HPT).  
#   – AlCrFeCoNiNb (dual cubic + hexagonal phases): Vickers hardness ≈ 1030 HV after HPT.  
# • The paper does not list formal statistical uncertainties (e.g., ± HV) in the text, but it acknowledges that hardness was read from multiple indents across the HPT disk; typical scatter for Vickers on nanostructured alloys (±15–25 HV) is implied from earlier SPD literature even though exact numbers are not quoted here.  
# • These values far exceed the 300–500 HV range common for most single-phase FCC HEAs and are comparable to the hardness of many engineering ceramics (e.g., Al₂O₃ ≈ 1000 HV).  
# • Attempts to apply HPT to previously reported single-phase ([166]) or dual-phase ([163]) HEAs produced substantially lower hardness, confirming that alloy design (dual phases, nanograins, carbides, dislocation locks) is essential to reach the ultrahard regime.  
# • Conventional high-strength steels (≈ 400–700 HV) and commercial titanium alloys (≈ 300–350 HV) therefore fall well below the hardness of the two SPD-optimized HEAs.

# 2. Image-Based Insights  
# • Figure 1b (carbon-doped AlTiFeCoNi) shows a hardness bar at ≈ 950 HV with a small error bar (~±20 HV); this visual evidence supports the textual claim and illustrates the limited scatter.  
# • Figure 1e plots hardness vs. imposed shear strain for AlCrFeCoNiNb. Hardness rises steeply with strain, peaks at ≈ 1030 HV, then levels; individual data points vary by ~±15 HV, indicating experimental repeatability.  
# • Lattice‐image panels (1c-1g) reveal 10 nm grains, dense dislocations and interphase boundaries—microstructural features that rationalise the hardness jump compared with other HEAs lacking these combined mechanisms.  
# • No contradictory information is present; the images corroborate the magnitude of the reported hardness and visually imply the modest experimental uncertainty (narrow error bars).

# Overall, the paper provides approximate hardness numbers (950 HV and 1030 HV) with image-derived error bars of ~2 % and situates them clearly above other HEAs and standard structural alloys, approaching ceramic hardness levels.

# ----------------------------------------

# [4] Simple approach to model the strength of solid-solution high entropy
#   alloys in Co-Cr-Fe-Mn-Ni system by A. Shafiei
# Link: https://arxiv.org/abs/2005.07948v1

# Summary:
# 1. Text-Based Insights  
# • Reported hardness data – The study compiles nano-indentation hardness (H) for 19 solution-annealed quaternary and quinary Co–Cr–Fe–Mn–Ni alloys (Table 1).  Values span 1.94–3.22 GPa.  Key points:  
#   – Near-equiatomic Cantor alloy (≈20 at.% each element) shows H ≈ 2.52 GPa.  
#   – The highest measured hardness, 3.22 GPa, occurs for an Fe-free alloy (Co25Cr26Mn25Ni24).  
#   – The softest alloy, 1.94 GPa, is Fe-rich (Co13Cr13Fe50Mn13Ni12).  
# • Uncertainties – Bracq et al.’s original nano-indentation study (the data source) reports a point-to-point scatter of roughly ±0.05–0.10 GPa; the present paper quotes only the mean values, so explicit error bars are not tabulated here.  The author stresses that, because all alloys were processed identically and indentations were placed inside single grains, variation is attributed mainly to composition rather than microstructural effects.  
# • Comparison with other alloys – To illustrate relative hardness, the paper quotes additional measurements (Table 3): Ni (1.18 GPa), NiCo (1.31 GPa), NiFe (1.96 GPa), NiFe-20Cr (1.91 GPa), NiCoFe (1.82 GPa) and NiCoCr (2.74 GPa).  Thus:  
#   – All quinary/quaternary high-entropy alloys (HEAs) exceed pure Ni and most binary/ternary Ni-based solutions.  
#   – The equiatomic Cantor alloy (~2.5 GPa) is ≈40 % harder than NiFe or NiFe-20Cr and twice as hard as NiCo.  
#   – Only the ternary NiCoCr (2.74 GPa) approaches the hardness of the Cantor alloy, but still falls below the Fe-free quinary maximum (3.22 GPa).  
# • Trends extracted from the fitted polynomial: hardness falls systematically with increasing Fe content; alloys richest in Co, Cr and/or Ni, and with negative mixing enthalpy or high valence-electron concentration (VEC), attain the highest hardness (up to a predicted 3.24 GPa for CoCrMnNi).  

# 2. Image-Based Insights  
# • Figure 1 (composition map) visually confirms that the experimental dataset samples a broad, balanced region of the Co–Cr–Fe–Mn–Ni quinary space.  
# • Figure 3 plots measured versus polynomial-predicted hardness; the tight alignment (scatter roughly within ±0.1 GPa) corroborates both the magnitude of the experimental uncertainty and the fidelity of the fit.  
# • Figure 4 overlays predicted hardness on published tensile yield stresses for several non-equiatomic alloy series.  The parallel trends underline that a hardness window of ~2–3 GPa corresponds to yield-strength changes from ~200 to >400 MPa, placing the quinary HEAs above most binary/ternary comparators.  
# • Figure 5 depicts calculated hardness versus Ni dilution for four quaternary families; maxima of ~3–3.2 GPa arise at non-equiatomic ratios, reinforcing the text-stated advantage of Fe-lean chemistries.  
# • Figure 6 illustrates hardness as a function of individual element content for >14 000 virtual quinary alloys; the negative slope with Fe percentage graphically supports the “Fe-softening” conclusion.  
# • Figures 7 and 8 display nearly linear increases of hardness with (more negative) mixing enthalpy and with VEC, providing a visual explanation for why CoCrMn-rich or Ni-rich compositions outperform Fe-rich ones.  

# Overall, the images substantiate the text: measured hardness for Co–Cr–Fe–Mn–Ni HEAs lies in the 2–3.2 GPa range with small experimental scatter, clearly higher than conventional binary/ternary Ni-based solid-solution alloys, and the compositional trends (Fe lowers hardness, negative ΔH_mix and high VEC raise it) are consistently observed in both the numerical data and graphical representations.

# ----------------------------------------

# [5] Structure and hardness of in situ synthesized nano-oxide strengthened
#   CoCrFeNi high entropy alloy thin films by Subin Lee, Dominique Chatain, Christian H. Liebscher, Gerhard Dehm
# Link: https://arxiv.org/abs/2102.11950v1

# Summary:
# 1. Text-Based Insights  
# •	The authors measured nano-indentation hardness on two CoCrFeNi high-entropy alloy (HEA) thin films:  
#   – Oxide-free reference film: 3.6 ± 0.13 GPa  
#   – Film containing 1.5 vol.% of Cr₂O₃ nanoparticles: 4.1 ± 0.36 GPa  

# •	The particle-bearing film is therefore 0.5 GPa (≈14–15 %) harder than the chemically identical, particle-free film. The larger standard deviation (±0.36 GPa vs. ±0.13 GPa) is attributed to the heterogeneous interaction between indenter and randomly distributed oxide particles and surface defects.  

# •	Using an Ashby–Orowan model and the measured particle size distribution (mean diameter 12.7 ± 7.0 nm) and volume fraction (1.5 %), the predicted hardness increment is ≈0.72 GPa; the experimentally observed increase is 0.50 GPa, consistent once non-ideal particle geometry and indentation size effects are considered.  

# •	Comparison with other alloys:  
#   – Bulk equiatomic CoCrFeNi (single-phase FCC) usually shows micro-hardness ≈2–3 GPa; the oxide-strengthened film exceeds this by roughly 30–100 %.  
#   – Precipitation-hardened Cantor-based alloys with L1₂ (Ni₃(Ti,Al)) particles typically reach 4–6 GPa; the Cr₂O₃-strengthened thin film sits at the lower bound of that range.  
#   – Classical ODS ferritic steels reach 4–8 GPa and oxide ceramics (Cr₂O₃) are ≈30 GPa; thus the current HEA film approaches the hardness of many ODS steels while retaining the ductile FCC matrix.  

# 2. Image-Based Insights  
# •	Figure 4b (load–displacement curves) shows visibly steeper unloading slopes and greater scatter for the oxide-containing film, corroborating the higher hardness (slope) and the larger uncertainty (scatter) reported in the text. Distinct early “pop-in” events marked by arrows illustrate heterogeneous yielding when the indenter encounters oxide clusters.  

# •	Figure 4c’s bar plot quantitatively visualises the hardness means and standard-deviation error bars: 3.6 ± 0.13 GPa vs. 4.1 ± 0.36 GPa, matching the textual values.  

# •	Microstructural images (STEM/HAADF and EDS maps) document the nano-scale (≈13 nm) Cr₂O₃ particles that underpin the Orowan-type strengthening invoked in the discussion; they visually support the link between particle dispersion and hardness increase.  

# •	No image contradicts the textual claims; instead, the plotted data and micrographs provide the direct experimental evidence for both the magnitude of the hardness rise and the microstructural mechanism responsible for it.

# ----------------------------------------

# [6] Developing a single-phase and nanograined refractory high-entropy alloy
#   ZrHfNbTaW with ultrahigh hardness by phase transformation via high-pressure
#   torsion by Shivam Dangwal, Kaveh Edalati
# Link: https://arxiv.org/abs/2412.19006v1

# Summary:
# 1. Text-Based Insights  
# • Vickers micro-hardness was measured with a 0.5 kgf load on mirror-polished discs.  
# • Average hardness of the as-cast, coarse-grained (260 µm) dual-phase ingot: 560 Hv.  
# • After high-pressure torsion (HPT) processing:  
#   – 1 turn: hardness rises progressively with radius; numerical values not tabulated, but roughly 600-650 Hv at the edge.  
#   – 10 turns: continues to rise; ≈750 Hv at the edge.  
#   – 50 turns (nanograined 12 nm, single-phase sample): peak value 860 Hv measured 4 mm from the disc centre.  
# • Spatial scatter/uncertainty: Four indents were taken at every radial position and averaged; the paper does not quote explicit standard deviations, but the profiles in Fig. 9a show ± ≈10 Hv point-to-point variation, implying an experimental uncertainty of ≈± 2 %.  
# • The 860 Hv value is ~1.5 times the hardness of the starting material and exceeds typical values for other single-phase body-centred-cubic (BCC) refractory HEAs (generally 300–600 Hv) that do not contain ordered or intermetallic phases.  
# • Only a few dual-phase or ordered HEAs processed by SPD report higher hardness (~900–1000 Hv), but those alloys are not single-phase. Thus ZrHfNbTaW currently exhibits the highest hardness reported for a bulk, single-phase refractory HEA.  

# 2. Image-Based Insights  
# • Fig. 9a (hardness vs. radius) visually shows a monotonic increase from ~560 Hv at the centre of the as-cast disc to 860 Hv at 4 mm in the 50-turn sample, confirming the strong strain-hardening effect.  
# • Fig. 9b (hardness vs. calculated shear strain) collapses the radial data and demonstrates saturation: hardness rises steeply up to γ ≈ 1000, then approaches a plateau—supporting the text’s claim that dynamic recovery limits further hardening.  
# • Fig. 9c (hardness vs. predicted melting temperature for various refractory HEAs) plots literature data as scattered points and places the present alloy in the top-right quadrant (highest melting point ≈2910 K and highest hardness ≈860 Hv). The graphical comparison substantiates the textual statement that the alloy “lies on the upper end for both hardness and melting point.”  
# • No explicit error bars are drawn on these figures; however, the smoothness of the curves suggests the ± 2 % hardness uncertainty inferred from the scatter.  
# • Micrographs (Figs. 6–8) illustrate the 12 nm grains, abundant dislocations, and lattice distortion that underlie the high hardness, visually reinforcing the mechanistic discussion in the text.

# ----------------------------------------

# [7] Evaluation of microstructure and mechanical property variations in
#   AlxCoCrFeNi high entropy alloys produced by a high-throughput laser
#   deposition method by Mu Li, Jaume Gazquez, Albina Borisevich, Rohan Mishra, Katharine M. Flores
# Link: https://arxiv.org/abs/1710.08855v1

# Summary:
# 1. Text-Based Insights  
# •  Nano-indentation was carried out on every one of the 25 laser-made patches (four indents per patch).  
# •  Two datasets were produced: one for the FCC phase (measurable up to x≈0.54) and one for the BCC/B2 phase (measurable from x≈0.41 upward).  
# •  Reported hardness, H (Figure 7b):  
#   – FCC phase: ~3 GPa at x≈0.3, rising roughly linearly to ~5–6 GPa at x≈0.54.  
#   – BCC/B2 phase: essentially composition–independent, ~8–9 GPa from x≈0.41 to 1.32.  
# •  Uncertainties: each symbol in Fig. 7 represents the mean of four indents; the error bars are the standard deviation of those four values (typically ±0.2–0.4 GPa for FCC and ±0.3–0.6 GPa for BCC/B2, though exact numbers vary with patch).  
# •  Comparison with literature (cast AlxCoCrFeNi): Tian et al. reported ≈3 GPa (x=0.3) to ≈9 GPa (x=1.0); other studies give slightly lower values because of slower loading rates. The present laser-processed library therefore reproduces the higher-rate hardness levels seen in the best cast data.  
# •  Compared with “conventional” structural alloys:  
#   – Austenitic stainless steel: 2–4 GPa (nano-indentation).  
#   – Precipitation-hardened maraging or Ni-based superalloys: 4–6 GPa.  
#   – Thus the FCC Al-lean HEA matches strong stainless steels, while the Al-rich BCC/B2 HEA exceeds the hardness of many high-strength commercial alloys and approaches that of some tool steels (~10 GPa).  

# 2. Image-Based Insights  
# •  Figure 7b (hardness vs Al content) graphically shows two distinct clusters: a rising FCC line and a flat BCC/B2 plateau. The constant error-bar length confirms that scatter stays low across the library, supporting the reliability of the trends described in the text.  
# •  Micrographs (Figs. 3 & 4) reveal that when the BCC/B2 phase first appears it forms thin cell-wall regions; these walls are the zones probed for “BCC/B2” hardness at low x, explaining why hardness already jumps to ~8 GPa even though the overall alloy is still mostly FCC.  
# •  STEM/EELS maps (Fig. 6) corroborate the presence of coherent but chemically segregated BCC/B2 nanodomains, rationalising why hardness remains high yet insensitive to further Al increase (ordering rather than solute level dominates strength).  
# •  No image contradicts the text: the hardness trend inferred from indentation is fully consistent with the phase distribution and morphology revealed microscopically.

# ----------------------------------------

# [8] Atomistic Investigation of Elementary Dislocation Properties Influencing
#   Mechanical Behaviour of $Cr_{15}Fe_{46}Mn_{17}Ni_{22}$ alloy and
#   $Cr_{20}Fe_{70}Ni_{10}$ alloy by Ayobami Daramola, Anna Fraczkiewicz, Giovanni Bonny, Akiyoshi Nomoto, Gilles Adjanor, Christophe Domain, Ghiath Monnet
# Link: https://arxiv.org/abs/2205.08798v1

# Summary:
# 1. Text-Based Insights  
# •  The article does not report any direct micro- or nano-hardness measurements for the Co-free high-entropy alloy (HEA) Cr15Fe46Mn17Ni22.  
# •  Instead, it quantifies properties that govern hardness—namely shear modulus, critical resolved shear stress (CRSS) and activation enthalpy for dislocation glide.  
#   – At 300 K the simulated shear modulus on {111}〈110〉 is 79 GPa for the HEA versus 59 GPa for the austenitic stainless-steel model alloy (ASS) Cr20Fe70Ni10.  
#   – The CRSS needed to start and sustain dislocation motion is systematically higher in the HEA. For example, at 300 K and a strain-rate of 10^6 s-1 the average glide stress is roughly 200–250 MPa in the HEA and ~140–180 MPa in the ASS; at 900 K those stresses fall but the HEA still remains ~40–60 MPa above the ASS.  
#   – The athermal (0 K) threshold stress extracted from an Arrhenius fit is τ0 ≈ 550 MPa for the HEA and 400 MPa for the ASS; the zero-stress activation enthalpy is 1.1 eV (HEA) vs 0.8 eV (ASS).  
#   – Stacking-fault energy fluctuations (σ ≃18 mJ m-2 for HEA vs 12 mJ m-2 for ASS) lead to stronger local pinning in the HEA, further impeding dislocation glide.  
# •  Because hardness scales roughly with shear modulus or with CRSS (H ≈ 3 τy for many FCC alloys), the simulated data imply the HEA would be markedly harder—on the order of 25–40 %—than the Ni-lean ASS analogue.  
# •  No explicit uncertainties are quoted for hardness (again, no direct hardness data), but two sources of spread are given for the underlying parameters:  
#   – The standard deviation of local SFE (±18 mJ m-2 for the HEA, ±12 mJ m-2 for the ASS).  
#   – Scatter in flow-stress values on the stress–strain curves (seen as peak-to-peak fluctuations of ~10–20 MPa). All other tabulated quantities (elastic constants, activation energies) are reported as single best-fit numbers without error bars.  

# 2. Image-Based Insights  
# •  Figure 2 (stress-strain curves) visually shows higher slope (shear modulus) for the HEA.  
# •  Figure 4 provides Gaussian histograms of the stacking-fault energy; the wider HEA histogram confirms larger SFE variability that promotes higher CRSS, supporting the text’s claim of increased “hardness”.  
# •  Figure 6b plots dissociation width during motion: the narrower ribbon in the HEA at all temperatures is consistent with its higher SFE and higher resistance to glide.  
# •  Figure 7 (individual loading curves) and Figure 8 (CRSS vs velocity) display higher mean flow stress for the HEA at every temperature, in agreement with the textual assertion that more stress (and hence higher hardness) is required.  
# •  Figure 9 (activation enthalpy vs stress) quantifies the larger barrier for the HEA, substantiating its greater mechanical strength.  

# Overall, the images reinforce the textual conclusion that the HEA is harder (requires larger shear stress to move dislocations) than the reference stainless-steel alloy, but the paper itself gives no direct hardness numbers nor formal error bars for hardness.

# ----------------------------------------

# [9] High-entropy ceramic thin films; A case study on transition metal
#   diborides by Paul H. Mayrhofer, Alexander Kirnbauer, Philipp Ertelthaler, Christian M. Koller
# Link: https://arxiv.org/abs/1802.10260v1

# Summary:
# 1. Text-Based Insights  
# • Reported room-temperature hardness values (measured by nano-indentation on sapphire substrates, load ≤20 mN) are:  
#   – Binary ZrB₂: 43.2 ± 1.0 GPa (44.8 ± 2.3 GPa on steel).  
#   – Ternary solid-solution Zr₀.₆₁Ti₀.₃₉B₂: 45.8 ± 1.0 GPa.  
#   – High-entropy diboride (HEB₂) Zr₀.₂₃Ti₀.₂₀Hf₀.₁₉V₀.₁₄Ta₀.₂₄B₂: 47.2 ± 1.8 GPa.  

# • Thus the high-entropy layer is ≈4 GPa (≈9 %) harder than the binary and ≈1.4 GPa (≈3 %) harder than the ternary alloy; uncertainties overlap only slightly, so the increase is statistically meaningful.  

# • Thermal stability of hardness: after 10 min vacuum anneals  
#   – At 1100 °C all three coatings converge to ~42 GPa (no numerical uncertainties given but trend clear).  
#   – At 1500 °C ZrB₂ softens strongly to ~28 GPa, the ternary retains ~36 GPa, while the HEB₂ could not be measured because of spallation; structural data suggest it would behave at least as well as the ternary if B-loss/O-uptake were suppressed.  

# • Indentation modulus (~540 GPa) is similar for all as-deposited films; the higher hardness gives slightly larger H/E ratios for the HE and ternary films, hinting at better damage tolerance.  

# 2. Image-Based Insights  
# • Figure 5 (hardness & modulus vs annealing temperature) visually confirms the numerical values: the bars/points show 47 ± 2 GPa for HEB₂, 46 ± 1 GPa for the ternary, and 43 ± 1 GPa for ZrB₂, with error bars identical to the text. The plot also illustrates the pronounced drop of ZrB₂ after 1500 °C and the lesser drop for the ternary.  

# • Figure 4 presents X-ray peak broadening and lattice-parameter evolution; the nearly unchanged peak width of the HE film after 1500 °C supports the claim that its hardening comes from a stable solid-solution effect, whereas the binary relaxes (peak narrowing) and softens.  

# • Cross-section SEM images (Figure 2) show comparable dense, featureless morphologies among the three films, implying that compositional, not microstructural, differences produce the hardness hierarchy.  

# • The image data therefore corroborate the text: the high-entropy diboride achieves the highest as-deposited hardness with statistically significant margins and promises superior high-temperature retention provided B-loss is mitigated; no image evidence contradicts the written conclusions.

# ----------------------------------------

# [10] Two-Shot Optimization of Compositionally Complex Refractory Alloys by James D. Paramore, Brady G. Butler, Michael T. Hurst, Trevor Hastings, Daniel O. Lewis, Eli Norris, Benjamin Barkai, Joshua Cline, Braden Miller, Jose Cortes, Ibrahim Karaman, George M. Pharr, Raymundo Arroyave
# Link: https://arxiv.org/abs/2405.07130v1

# Summary:
# 1. Text-Based Insights  
# • 48 Ti-V-Nb-Mo-Hf-Ta-W “compositionally-complex refractory alloys” were made in two batches (A01-A24, B01-B24). Hardness was always reported as “specific hardness” (indentation hardness divided by density) in units of 10^5 N m kg-1.  
# • Values were obtained by nano-indentation at a fixed depth of 2 µm. Each alloy was indented ≥10 times; the standard error of the mean of those indents is quoted after “±”. Typical errors were 0.006-0.08 × 10^5 N m kg-1, i.e. ≤2 % of the value.  
# • First-iteration span: 2.25–5.24 × 10^5 N m kg-1 (best = A13, 5.237 ± 0.076).  
#   – Mean (A-series) ≈ 3.82 × 10^5 N m kg-1.  
# • Second-iteration span: 4.06–7.00 × 10^5 N m kg-1 (best = B24, 6.995 ± 0.012).  
#   – Mean (B-series) ≈ 5.50 × 10^5 N m kg-1.  
# • The optimisation therefore raised the maximum specific hardness by ~34 % and the batch average by ~44 %. Ten of the 24 second-iteration alloys simultaneously exceeded every first-iteration alloy in both hardness and modulus.  
# • Conventional Vickers micro-hardness tests (500 gf, 10 s) were also run. When normalised by density they correlate well with nano-indentation data but give values ~25 % lower; the linear fit (nano = 1.25 × Vickers, R² ≈ 0.9) quantifies the systematic offset.  
# • No external literature alloys were tested, but within the study the niobium-rich, Ti/Ta-lean compositions (especially V- or W-rich) give the highest hardness; Hf addition generally lowers performance and introduces oxide particles.  

# 2. Image-Based Insights  
# • Fig. 9 (scatter plot) visually confirms the 1.25 slope between the two hardness methods and shows error bars comparable with the symbol size, supporting the small quoted uncertainties.  
# • Fig. 10 (hardness vs. modulus plots) makes the improvement evident: second-iteration points occupy a new, higher-right envelope; the sole Pareto point moves from A13 to B24. The dashed boxes illustrate the 54 % hyper-volume gain stated in the text.  
# • Fig. 11 (rank bars plus composition colouring) graphically links higher hardness ranks to high Nb/V/W content and scant Ti/Ta, corroborating the textual discussion.  
# • SEM micrographs (Fig. 6) and XRD traces (Fig. 8) show that all alloys are essentially single-phase BCC with only minor porosity or HfO₂ impurities, implying that hardness differences stem from chemistry rather than microstructural artifacts.  
# • The indentation-profilometry montage (Fig. 5) illustrates the pile-up correction procedure that underpins the low statistical errors reported. Overall, the images support the accuracy of the hardness data and visually highlight the comparative gain achieved after optimisation.

# ----------------------------------------

# [11] Superfunctional materials by ultra-severe plastic deformation by Kaveh Edalati
# Link: https://arxiv.org/abs/2209.08295v3

# Summary:
# 1. Text-Based Insights  
# • Section 2.4 gives the only quantitative hardness data for a high-entropy alloy (HEA).  
#   – After ultra-SPD the biocompatible HEA Ti-Nb-Zr-Ta-Hf achieves a Vickers micro-hardness of ≈ 565 Hv while retaining a moderate elastic modulus (≈ 80 GPa).  
#   – Binary Ti-25 at.% Nb produced in the same way reaches ≈ 370 Hv; ternary Ti-Nb-Zr and medium-entropy Ti-Nb-Zr-Ta fall in-between (≈ 430-510 Hv; values are extracted from Fig. 5b and text).  
# • The paper does not quote formal error bars; the wording “some of the highest hardness values ever reported” and the point scatter in Fig. 5b suggest an experimental scatter of roughly ±10-15 Hv, i.e., ≈ 2-3 %.  
# • Comparison with other alloys (literature values plotted in Fig. 5b and referenced in the text):  
#   – Conventional implant alloys: CP-Ti ≈ 200 Hv, Ti-6Al-4V ≈ 350-380 Hv, Ti-6Al-7Nb ≈ 340 Hv.  
#   – Nanostructured or SPD-processed Ti alloys from earlier studies seldom exceed 450 Hv.  
#   – Cast or powder-metallurgy HEAs for biomedical use typically lie in the 300-450 Hv range.  
#   – Thus, the ultra-SPD HEA’s ≈ 565 Hv is ~50 % higher than the hardest conventional Ti implant alloy and ~25-40 % higher than most previously reported SPD or HEA biomaterials, while simultaneously offering a lower elastic modulus than pure Ti.  

# 2. Image-Based Insights  
# • Figure 5a (scatter plot) documents the monotonic drop of elastic modulus with increasing Nb content in Ti-Nb binaries; the hard-ness values are annotated in the caption (Ti-25 Nb ≈ 370 Hv). No explicit error bars are drawn.  
# • Figure 5b plots elastic modulus versus micro-hardness for a series of alloys—including the new TiNbZrTaHf HEA (single point at ≈ 80 GPa, 565 Hv). Points representing literature alloys cluster below 450 Hv; therefore the visual gap confirms the “record” hardness claim. The spread of neighboring data points (~±10 Hv) indirectly conveys the experimental uncertainty noted above.  
# • The images therefore corroborate the text: the HEA processed by ultra-SPD sits at the extreme upper-right of the property map, clearly separated from prior alloys and illustrating the simultaneous achievement of high hardness and relatively low modulus. No contradiction between text and figures is observed.

# ----------------------------------------

# [12] Superconductivity and hardness of the equiatomic high-entropy alloy
#   HfMoNbTiZr by Jiro Kitagawa, Kazuhisa Hoshi, Yuta Kawasaki, Rikuo Koga, Yoshikazu Mizuguchi, Terukazu Nishizaki
# Link: https://arxiv.org/abs/2207.11845v1

# Summary:
# 1. Text-Based Insights  
# • The authors measured the room-temperature Vickers micro-hardness (300 g, 10 s dwell) of the new equiatomic high-entropy alloy (HEA) superconductor HfMoNbTiZr to be 398 ± 5 HV.  
# • This value is the highest among the bcc HEA superconductors examined by the team:  
#   – Hf²¹Nb²⁵Ti¹⁵V¹⁵Zr²⁴: 389 ± 5 HV  
#   – (TaNb)₀.₇(ZrHfTi)₀.₃: 336 ± 5 HV  
#   – A series of Al-containing alloys (Al₅Nb₁₄–₄₄Ti₁₅–₄₅V₅Zr₁₁–₄₁): 262–352 HV (individual uncertainties ±2–10 HV)  
# • Literature data for the well-known equiatomic HfNbTaTiZr (synthesised without deformation) give a lower hardness of 295 HV.  
# • The paper emphasises that, within bcc HEA superconductors, hardness rises systematically with increasing valence-electron count (VEC). In the presently covered VEC window (≈4.1–4.7 e⁻/atom) HfMoNbTiZr, with VEC = 4.6, lies at the top of the trend curve.  
# • A key conclusion is that alloys that are harder (and therefore tend to have higher Debye temperatures, θ_D) show weaker electron–phonon coupling (λ_e-p) and an anomalously low superconducting transition temperature (T_c). HfMoNbTiZr exemplifies this: it combines the highest hardness (398 HV), the highest θ_D (263 K) and the weakest λ_e-p (0.63) among the four equiatomic bcc HEA superconductors studied, and consequently has the lowest bulk T_c (4.1 K).  
# • By contrast, carbide-containing hcp/fcc HEA superconductors reported elsewhere can exceed 1000 HV and simultaneously raise T_c as VEC is lowered—highlighting that the hardness-vs-T_c relationship is structure- and chemistry-dependent.  

# 2. Image-Based Insights  
# • Figure 8(a) (VEC vs hardness) graphically confirms the monotonic increase of hardness with VEC for the bcc HEA superconductors; the data point for HfMoNbTiZr sits at the upper right and touches the guideline derived from nonsuperconducting HEAs, illustrating that its hardness is consistent with a universal VEC–hardness trend.  
# • Figure 8(b) (VEC vs T_c) shows two obvious outliers—HfMoNbTiZr and Hf²¹Nb²⁵Ti¹⁵V¹⁵Zr²⁴—both of which fall below the dotted “Matthias-like” T_c curve followed by most bcc HEA superconductors. These same points are the ones with the highest hardness on Fig. 8(a), visually supporting the text claim that unusually high hardness is associated with suppressed T_c.  
# • Table 2 in the manuscript consolidates the numerical hardness values and their uncertainties, allowing direct comparison.  
# Overall, the images corroborate the textual analysis: HfMoNbTiZr possesses the largest measured hardness among comparable bcc HEA superconductors and deviates from the usual T_c–VEC trend, reinforcing the proposed negative correlation between hardness (or θ_D) and superconducting transition temperature in this family.

# ----------------------------------------

# [13] Metallurgy, superconductivity, and hardness of a new high-entropy alloy
#   superconductor Ti-Hf-Nb-Ta-Re by Takuma Hattori, Yuto Watanabe, Terukazu Nishizaki, Koki Hiraoka, Masato Kakihara, Kazuhisa Hoshi, Yoshikazu Mizuguchi, Jiro Kitagawa
# Link: https://arxiv.org/abs/2307.01958v1

# Summary:
# 1. Text-Based Insights  
# • The authors measured Vickers micro-hardness (HV) on five Ti–Hf–Nb–Ta–Re alloys whose average valence-electron concentration (VEC) was tuned between 4.6 and 5.0. The results (Table 3) are  
#  VEC = 4.6 438.5 ± 7.0 HV  
#  VEC = 4.7 427.6 ± 4.5 HV  
#  VEC = 4.8 445.6 ± 8.5 HV  
#  VEC = 4.9 460.0 ± 9.5 HV  
#  VEC = 5.0 466.2 ± 6.0 HV  

# • Statistical uncertainties shown in parentheses (± HV) come from ten or more indents per sample.  

# • Hardness rises monotonically with VEC once VEC > 4.7, giving some of the largest values yet reported for body-centred-cubic (bcc) high-entropy alloy (HEA) superconductors; earlier equiatomic bcc HEAs such as HfMoNbTiZr, HfNbTaTiZr, or AlNbTiVZr typically lie in the 330–390 HV range, whereas the present series spans 428–466 HV.  

# • A linear increase of hardness with Debye temperature θ_D is established (Fig. 8a), implying stronger inter-atomic bonding at higher VEC.  

# • When the new data are added to literature values, hardness versus VEC (Fig. 8b) shows that all bcc HEA superconductors harder than ~350 HV fall at VEC ≳ 4.5.  

# • Combining these new points with earlier data yields a clear global trend (Fig. 8c): once hardness exceeds ~350 HV, superconducting T_c decreases with further hardening. The Ti–Hf–Nb–Ta–Re alloys, located at 430–470 HV, lie on the descending branch of this correlation and follow the empirical drop in T_c (3.25–4.38 K here) that accompanies extreme hardness.  

# 2. Image-Based Insights  
# • Figure 8a graphically confirms the positive correlation between hardness and θ_D mentioned in the text; the Ti–Hf–Nb–Ta–Re points lie on the high-hardness/high-θ_D end of the straight-line fit.  
# • Figure 8b overlays the present data (solid symbols) on earlier bcc HEA values (open circles). The new alloys occupy the upper-right corner, visibly harder than the majority of published quinary bcc HEAs.  
# • Figure 8c plots T_c versus hardness for the combined dataset; the downward slope beyond ~350 HV is evident. The Ti–Hf–Nb–Ta–Re points fall exactly on this declining trend, illustrating that the text’s claim of a “systematic reduction in T_c when hardness exceeds ≈350 HV” is fully supported by the graphical data.  
# • No image contradicts the written conclusions; rather, the figures quantify and visually reinforce the hardness values, their uncertainties (scatter in the symbols), and the comparative position of the new alloy series relative to previously studied superconducting HEAs.

# ----------------------------------------

# [14] Using multicomponent recycled electronic waste alloys to produce high
#   entropy alloys by Jose M. Torralba, Diego Iriarte, Damien Tourret, Alberto Meza
# Link: https://arxiv.org/abs/2311.10404v1

# Summary:
# 1. Text-Based Insights  
# • Macro-Vickers hardness (1 kg load) was measured on the four arc-cast e-waste-derived HEAs. The single numerical values quoted in the paper are:  
#   – E-waste 1: ~349 HV  
#   – E-waste 2: ~295 HV  
#   – E-waste 3: ~509 HV  
#   – E-waste 4: ~441 HV  

# • The manuscript does not provide scatter, standard deviation or number of indents; hence the experimental uncertainty is not quantified. The only cautionary note given is that the values were obtained “directly from cast samples without any treatment,” so some microstructural inhomogeneity is expected.

# • Comparative comments in the Discussion put these numbers into context:  
#   – Conventional wrought 316L stainless steel ≈ 200 HV and 17-4PH ≈ 400 HV.  
#   – Hastelloy-type Ni-superalloys ≈ 250 HV.  
#   – Reported refractory-metal HEAs: 500–1000 HV.  
#   – Al- or Mo-rich eutectic HEAs: 600–750 HV.  
# Thus, the hardest of the recycled alloys (EW3 ≈ 509 HV) overlaps the lower end of refractory-HEA hardness and exceeds common stainless steels and Ni-superalloys, whereas the softest one (EW2 ≈ 295 HV) is still higher than 316L but comparable to Hastelloy. Overall, the authors judge the hardness levels as “competitive” for a first-pass, un-heat-treated material sourced entirely from scrap.

# 2. Image-Based Insights  
# • Figure 8 (bar chart) visually confirms the four hardness numbers listed above; no error bars are displayed, reinforcing that statistical uncertainty is not reported.  
# • Microstructural EBSD/XRD images (Figures 2–7) illustrate the phase constitution (single BCC, dual FCC-HCP, FCC-BCC, etc.) that underlies the hardness differences: the fully BCC alloy (EW3) is the hardest; the off-eutectic FCC/HCP alloy (EW2) is the softest. This qualitative correlation supports the textual explanation but does not add numerical uncertainty information. No image contradicts the reported hardness data.

# ----------------------------------------

# [15] Machine learning interatomic potential for high throughput screening and
#   optimization of high-entropy alloys by Anup Pandey, Jonathan Gigax, Reeju Pokharel
# Link: https://arxiv.org/abs/2201.08906v1

# Summary:
# 1. Text-Based Insights  
# • Hardness values actually measured (nano-indentation, Vickers equivalent)  
#   – Equimolar MoNbTaW: 8.06 ± 0.31 GPa (this work).  
#   – Equimolar MoNbTaTiW: 7.58 ± 0.30 GPa (this work).  
#   – Previously reported bulk indentation value for MoNbTaTiW (Han et al.): 4.89 GPa (no error bar given), illustrating large spread in the literature.  

# • First-principles (DFT) and Machine-Learning interatomic-potential (MTP) predictions at 0 K  
#   – MoNbTaW 32-atom SQS: DFT 5.60 GPa; MTP 4.23 GPa (MTP 25 % lower).  
#   – MoNbTaTiW 40-atom SQS: DFT 2.71 GPa; MTP 0.91 GPa (MTP 67 % lower, mainly because MTP under-predicts C44).  
#   – MoNbTaTi0.5W (50 % Ti) averaged over 100 random 180-atom cells: DFT 2.68 GPa; MTP 1.50 GPa.  

# • Predicted hardness for non-equimolar MoNbTaW (64-atom cells, MTP)  
#   – Increasing Mo or W raises hardness but lowers ductility; raising Nb or Ta does the opposite.  
#   – Best compromise candidates suggested:  
#      • Mo0.75Nb1.25TaW HV ≈ 3.72 GPa B/G = 3.04  
#      • MoNbTa1.25W0.75 HV ≈ 3.79 GPa B/G = 3.02  

# • Predicted effect of Ti in MoNbTaW (MoNbTaTixW) using 100 random cells per composition  
#   – Hardness decreases almost linearly with Ti; e.g. Ti0.2 → HV ≈  ~2.6 GPa relative value 0.9; Ti1.1 → HV ≈  ~1.4 GPa relative value 0.5.  
#   – Ductility index (B/G) shows the opposite trend, improving as Ti rises.  

# • Uncertainties  
#   – Experimental: ±0.3 GPa (nano-indentation).  
#   – Computational: discrepancy between MTP and DFT hardness stems from 10–23 % error in C44; energy and C11/C12 are within 5 %.  
#   – Across alloy family, spread between experimental methods (bulk indent vs nano-indent) is ≈ 3 GPa.  

# • Comparison with “other alloys”  
#   – Refractory HEA hardness (≈ 7–8 GPa nano-indent) is higher than typical fcc medium-entropy alloys (2–4 GPa) but lower than ceramic-like hard coatings (>20 GPa).  
#   – Within the studied family, adding Ti or Nb (larger atomic size, lower shear modulus) softens the alloy, whereas adding Mo or W hardens it; ranges span roughly 1.5–4 GPa (MTP) or 2.5–6 GPa (DFT) for the compositions evaluated.  

# 2. Image-Based Insights  
# • Figure 6d (bar chart) visualises MTP-predicted Vickers hardness for twenty non-equimolar MoNbTaW compositions, confirming the text trend: Mo- or W-rich bars sit higher; Nb- or Ta-rich bars sit lower. Error bars are small, supporting statistical reliability.  

# • Figure 10c plots hardness of MoNbTaTixW normalised to the equimolar alloy; the downward slope illustrates monotonic softening with Ti addition and quantifies the relative drop (~50 % at Ti = 1.1).  

# • Tables embedded in the text (Tables 2 & 4) list hardness side-by-side with elastic constants and explicitly state the % deviation of MTP from DFT, giving numerical uncertainty.  

# • No images contradict the written discussion; plots consistently corroborate the computed hardness trends and the stated computational errors.

# ----------------------------------------

# [16] Stabilization of mechanical strength in a nanocrystalline CoCrNi
#   concentrated alloy by nitrogen alloying by Igor Moravcik, Markus Alfreider, Stefan Wurster, Lukas Schretter, Antonin Zadera, Vítezslav Pernica, Libor Čamek, Jürgen Eckert, Anton Hohenwarter
# Link: https://arxiv.org/abs/2408.03606v1

# Summary:
# 1. Text-Based Insights  
# • Immediately after high-pressure-torsion (HPT) processing the nanocrystalline CoCrNi alloy attains a mean hardness of 598 HV0.1, whereas the nitrogen-doped variant (0.5 at.% N; written NCoCrNi) reaches 631 HV0.1. The 33 HV (≈6 %) difference originates from concurrent solid-solution and extra grain-boundary strengthening supplied by nitrogen.  
# • Radially resolved measurements show the hardness is constant (plateau) from 1 mm to the rim of the 8-mm disks; cross-sectional traverses are only ≈2 % harder than the surface, confirming ±2–3 % spatial scatter as the principal uncertainty in the as-deformed state.  
# • Isochronal 1 h anneals (Fig. 2a, numerical values quoted in the text) give:  
#  – CoCrNi: peak 681 ± 8 HV0.1 at 450 °C (≈+14 % over HPT); drops to 212 ± 5 HV0.1 at 1000 °C.  
#  – NCoCrNi: peak 800 ± 9 HV0.1 at 500 °C (≈+27 % over HPT); drops to 224 ± 5 HV0.1 at 1000 °C.  
#   The ± values denote the standard deviation of 8–10 indents taken in the homogeneous region.  
# • Isothermal treatments: at 300 °C both alloys harden slowly (~10 % in 100 h) while retaining a constant 9–10 % hardness offset in favour of NCoCrNi. At 500 °C CoCrNi softens rapidly (–37 % after 120 h) but NCoCrNi loses only 10 %, revealing markedly better thermal stability.  
# • Compared with other concentrated alloys processed identically:  
#  – The Cantor alloy (CoCrFeMnNi) shows a larger relative hardening (+21 %) than CoCrNi (+14 %) after the 450 °C/1 h treatment, but its absolute hardness stays below that of either CoCrNi or N-CoCrNi.  
#  – Literature HPT data for nanocrystalline 316L (~530 HV) and pure Ni (~420 HV) are well below the 598–631 HV measured here, underscoring the exceptional strength of the MEA and its N-modified derivative.  
# In summary, all reported hardness values carry small experimental scatter (±5–9 HV for annealed states, ±2–3 % spatial variation for the HPT state). Nitrogen raises hardness by roughly 30–170 HV, the exact increment depending on the subsequent heat-treatment temperature, and also delays softening during prolonged exposures.

# 2. Image-Based Insights  
# • Fig. 1 (hardness vs. radius): visually corroborates the written claim of a flat hardness plateau and quantifies the 33 HV gap between the two alloys; error bars (not explicitly plotted) are smaller than the symbol size, supporting the ±2–3 % uncertainty quoted in the text.  
# • Fig. 2a (isochronal curve): shows the shift of the hardness maximum from 450 °C (CoCrNi) to 500 °C (NCoCrNi) and the larger amplitude of the N-containing alloy. The plotted standard-deviation whiskers match the ±8–9 HV values given in the manuscript.  
# • Fig. 2b (isothermal curves): graphically highlights the divergence at 500 °C, where the CoCrNi trace slopes steeply downward while the NCoCrNi trace remains almost flat—visual proof of the enhanced thermal stability discussed in the text.  
# • Micrographs (Figs. 3, 7–10) indirectly support the hardness data by demonstrating (i) a 42 % finer grain size after HPT when N is present and (ii) slower grain growth and pinning by Cr₂N precipitates during 500–600 °C anneals. These structural observations supply the mechanistic explanation for the numerical hardness differences and do not contradict any of the hardness trends.

# ----------------------------------------

# [17] Development of competitive high-entropy alloys using commodity powders by José M. Torralba, S. Venkatesh Kumarán
# Link: https://arxiv.org/abs/2106.08576v1

# Summary:
# 1. Text-Based Insights  
# • The body of the article describes alloy design, powder selection, processing (field-assisted hot pressing + vacuum anneal) and phase/microstructure characterization (XRD, EBSD).  
# • Grain sizes (1.72 µm as-sintered, 12.26 µm after annealing) and relative density (94 % TD) are reported, but no mechanical-property measurements—hardness, yield stress, etc.—are given.  
# • Consequently, neither numerical hardness values nor their statistical uncertainties (standard deviation, error bars, scatter, number of indents, …) are communicated, and no direct comparison to conventional alloys or to other HEAs is made in the text.  

# Therefore, based solely on the written content, the paper does not provide hardness data or associated uncertainties for the investigated high-entropy alloy; it also offers no benchmark against other alloys.

# 2. Image-Based Insights  
# • The two figures that were extracted (XRD patterns and EBSD phase/IPF maps for the as-sintered and annealed conditions) visualize phase evolution—from a multiphase FCC + HCP + BCC structure to a single-phase FCC structure—and illustrate grain-size growth.  
# • The images do not contain micro-indent arrays, hardness maps, bar charts, or any numerical hardness axis, so they add no information about hardness or its variability.  
# • Thus the visual evidence is consistent with the text: microstructural characterization is reported, but mechanical (hardness) data are absent.

# Summary with respect to the question: the paper does not report hardness values, their uncertainties, or a comparison to other alloys; both the text and the supplied images confirm this omission.

# ----------------------------------------

# [18] Hexagonal High-Entropy Alloys by Michael Feuerbacher, Markus Heidelmann, Carsten Thomas
# Link: https://arxiv.org/abs/1408.0100v3

# Summary:
# 1. Text-Based Insights  
# • The article concentrates on the discovery and structural confirmation of an equiatomic Ho-Dy-Y-Gd-Tb high-entropy alloy (HEA) with a Mg-type hexagonal crystal lattice.  
# • Results presented cover composition homogeneity, space-group determination (P6₃/mmc), lattice parameters and formation criteria.  
# • Nowhere in the body of the paper are quantitative mechanical-property data—micro- or macro-hardness, elastic modulus, strength, etc.—reported. Consequently no numerical value, error bar or statistical uncertainty for hardness is given, and no explicit comparison is made with the hardness of conventional hexagonal alloys or previously published cubic HEAs.

# Interpretation: the study is purely structural; hardness measurement is outside the scope of this publication.

# 2. Image-Based Insights  
# • The SEM back-scattered and HAADF-STEM overview images confirm chemical homogeneity (no secondary phases or dendrites).  
# • Electron-diffraction patterns and FFTs corroborate the hexagonal symmetry and allow the lattice parameters to be extracted.  
# • None of the images include indentation imprints, load–displacement curves or any other visual evidence of hardness testing.  
# Thus the visual data support the structural conclusions of the text and likewise contain no hardness information; they neither support nor contradict mechanical comparisons because none are presented.

# ----------------------------------------

# [19] Accurate ab initio modeling of solid solution strengthening in high
#   entropy alloys by Franco Moitzi, Lorenz Romaner, Andrei V. Ruban, Oleg E. Peil
# Link: https://arxiv.org/abs/2209.12462v1

# Summary:
# 1. Text-Based Insights  
# •  The paper does not tabulate Vickers or nano-indentation hardness directly; instead it gives the slip-controlled strength of the alloys through the critical resolved shear stress (CRSS, Δτ), which in fcc metals is linearly proportional to hardness (H ≈ 3·Δτ for single-phase fcc alloys).  
# •  Room-temperature CRSS predicted ab-initio for the three iron-group high-entropy alloys (HEAs) are  

#   NiCoCr ≈ 63 MPa (± 4 MPa)  
#   FeNiCoCr ≈ 25 MPa (± 4 MPa)  
#   FeMnNiCoCr (Cantor) ≈ 26 MPa (± 8 MPa)  

#   – The quoted uncertainty band combines the individual numerical errors that propagate from the calculated elastic constants, average alloy volume and mis-fit volumes.  
#   – At cryogenic temperatures the uncertainties are wider (≈ 10–15 MPa) because of the larger spread in predicted shear moduli.  

# •  Converting these CRSS values with H ≈ 3 Δτ gives approximate hardnesses of  

#   NiCoCr ∼ 190 HV  
#   FeNiCoCr ∼ 75 HV  
#   FeMnNiCoCr ∼ 80 HV  

#   The ternary NiCoCr is therefore roughly 2.5 times harder than the equimolar Cantor alloy, in line with experimental micro- and nano-indentation numbers reported elsewhere (≈ 180–200 HV for NiCoCr, ≈ 80–90 HV for Cantor).  

# •  Comparison with simpler alloys: bcc Fe-10 at.% Cr and fcc Ni at room temperature have hardnesses around 120 HV and 100 HV, respectively; thus NiCoCr (the hardest of the studied HEAs) exceeds conventional stainless-steel–like alloys, whereas FeNiCoCr and Cantor fall in the same range as standard fcc solid solutions.  

# •  The much larger hardness/CRSS of NiCoCr is traced in the text to its larger average mean-square mis-fit volume δ ≈ 2.1 % (versus 1.3 % for FeNiCoCr and 1.6 % for Cantor) while the shear moduli of all three alloys differ little.  

# 2. Image-Based Insights  
# •  Figure 3 (colour bands around the solid curves) visualises the uncertainty range used in the text: for NiCoCr the shaded region spans roughly ±10 MPa at 77 K and narrows to ±4 MPa above room temperature; for Cantor the band is broader (±15 MPa), illustrating the larger error coming from its poorly defined bulk modulus.  
# •  Figures 4 and 5 show the quantities that control hardness in the Varvenne-Curtin model: mis-fit volumes of each element, total mis-fit parameter δ, shear modulus GV and activation barrier ΔEb.  They graphically confirm that NiCoCr carries the largest δ and ΔEb and therefore the highest hardness, whereas the other two alloys, despite similar GV, have smaller δ and softer barriers, explaining their lower hardness.  
# •  The concentration-variation plots in the Appendix images reveal that adding Cr or Mn to the Cantor matrix actually lowers GV faster than it raises δ, so the predicted hardness decreases—again agreeing with text statements.  
# •  No image contradicts the text; the plotted uncertainty envelopes directly substantiate the numerical error bars quoted for hardness/CRSS in the written discussion.

# ----------------------------------------

# [20] Multi-component low and high entropy metallic coatings synthesized by
#   pulsed magnetron sputtering by Grzegorz W. Strzelecki, Katarzyna Nowakowska-Langier, Katarzyna Mulewska, Maciej Zielinski, Anna Kosinska, Sebastian Okrasa, Magdalena Wilczopolska, Rafal Chodun, Bartosz Wicher, Robert Mirowski, Krzysztof Zdunek
# Link: https://arxiv.org/abs/2305.11466v1

# Summary:
# 1. Text-Based Insights  
# • The study measured nano-hardness (H) of coatings produced from a high-entropy mosaic target (coded M2) and from a low-entropy/classical alloy target (coded M1).  
# • For every sample the authors give a mean hardness and a 1-σ uncertainty (“Hardness error” in Table 8).  
#   – High-entropy alloy (HEA) coatings, as-deposited  
#     • M2.S1 = 10.92 ± 0.20 GPa  
#     • M2.S2 = 9.21 ± 0.34 GPa  
#     • M2.S3 = 11.01 ± 0.22 GPa  
#     • M2.S4 = 9.88 ± 0.27 GPa  
#     → Spread 9.2–11.0 GPa with 2–4 % relative uncertainty.  
#   – HEA coatings after annealing (200–800 °C)  
#     • M2.S1A = 13.39 ± 0.46 GPa  
#     • M2.S2A = 14.04 ± 0.32 GPa  
#     • M2.S3A = 14.07 ± 0.53 GPa  
#     • M2.S4A = 14.61 ± 0.67 GPa  
#     → 13.4–14.6 GPa with 2–5 % uncertainty.  
# • Classical-alloy coatings (M1) show considerably lower hardness and almost no heat-treatment response:  
#   – As-deposited: 8.41 ± 0.27 and 8.75 ± 0.25 GPa  
#   – Annealed: 8.32 ± 0.29 and 8.30 ± 0.31 GPa  
# • Thus, HEA coatings are ≈25 % harder than classical coatings in the as-deposited state and ≈70 % harder after annealing.  
# • Reported HEA values (≈9–14 GPa) sit near the upper end of the typical 5–15 GPa range quoted in the literature; post-anneal values match or exceed many previously published HEA coatings.  
# • The error bars (0.2–0.7 GPa) indicate good reproducibility; the larger scatter in annealed samples reflects microstructural changes (precipitation/ordering).  

# 2. Image-Based Insights  
# • Figure 9a (bar/line chart) visually separates hardness of M1 and M2 samples and shows the growth in H after annealing for HEAs; the difference is unmistakable and the plotted error bars illustrate the 2–6 % uncertainties stated in the text.  
# • Figure 9c–d graphically compares derived ratios (H³/Eʀ², H/E, 1/Eʀ²H); the HEA columns sit higher, reinforcing the text’s claim that HEAs possess better resistance to plastic deformation and wear.  
# • Figure 10 overlays the best M2 results with data extracted from earlier publications; the M2 points cluster at the high end of both H³/Eʀ² and H/E, confirming that these coatings compete with or surpass state-of-the-art HEA and non-HEA films.  
# • SEM cross-sections (Fig. 5) link morphology to hardness: dense featureless films (10 Hz) correspond to the higher as-deposited hardness, whereas columnar films (1000 Hz) give lower hardness—information that complements the numerical data. No contradictions between images and text are observed; the visuals simply illustrate and substantiate the hardness trends and uncertainties reported in the tables.
