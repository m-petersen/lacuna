# What is lesion network mapping?

Lacuna characterizes a lesion by contextualizing it against several kinds of normative
reference data. *Lesion network mapping* is one family of methods within that broader
characterization — the one concerned with brain **networks**. This page explains the idea
and how to choose among Lacuna's analyses.

A brain lesion rarely acts in isolation. Its consequences depend not only on the
tissue it destroys, but on the wider network that tissue belonged to. Two lesions
of similar size in different locations can produce very different symptoms, and
two lesions in *different* locations can produce *similar* symptoms — because they
sit on the same network. **Lesion network mapping** is the family of methods that
makes this network context explicit.

The core idea is to take an individual lesion mask and look it up in *normative*
brain data — connectivity measured in large samples of healthy people — to infer
which circuits the lesion engages. Because the connectivity comes from a normative
reference rather than the patient, the method works even when you only have a
structural lesion mask and no functional or diffusion imaging of the patient.

## The questions Lacuna can answer

Lacuna provides four analyses. They differ in the kind of "damage" they measure
and the data they require.

### Focal damage

The most direct question: **how much of each brain region does the lesion
destroy?** Lacuna overlays the lesion mask on one or more brain parcellation
atlases and reports the overlap per region. This needs no normative connectome —
the atlases ship with Lacuna — so it is the fastest analysis and a good starting
point.

Use it when you want anatomy-based damage scores, or as a parcel-level summary
alongside a network analysis.

### Functional lesion network mapping (FNM)

**Which functional circuit is connected to the lesion site?** Lacuna takes the
lesion mask as a seed and computes its resting-state functional connectivity to
the rest of the brain, using a normative functional connectome (e.g. GSP1000).
The result is a whole-brain map of the functional network linked to the lesion.

Use it when you care about functional circuitry — for example relating lesions to
behavioral or cognitive syndromes that are thought to be network-based.

### Structural lesion network mapping (SNM)

**Which white-matter connections does the lesion disconnect?** Instead of
functional correlation, SNM uses a normative tractogram (e.g. HCP1065 or dTOR985)
to find the streamlines that pass through the lesion, yielding a map of structural
disconnection. SNM additionally requires [MRtrix3](https://www.mrtrix.org/).

Use it when the question is about anatomical disconnection rather than functional
coupling.

### Accelerated functional network mapping (AFNM)

This answers the **same question as FNM** but with a faster, matrix-based
implementation built on a parcellated functional connectome. It trades the
full voxel-resolution map for substantially lower compute and memory cost.

Use it when you are processing many subjects and a parcel-resolution functional
map is sufficient; use the standard FNM when you need voxel-level detail.

## Choosing an analysis

| If you want to know… | Use | Normative data |
|---|---|---|
| How much each region is damaged | Focal damage | None (bundled atlases) |
| The functional network of the lesion | FNM | Functional connectome |
| The functional network, fast / at scale | AFNM | Parcellated functional connectome |
| Which tracts are disconnected | SNM | Tractogram (+ MRtrix3) |

These analyses are complementary, not mutually exclusive — a typical study runs
focal damage for interpretable anatomy plus one network analysis for the
circuit-level picture. All of them expect the lesion mask in a supported MNI space
(see [coordinate spaces](coordinate-spaces.md)); Lacuna aligns the normative data
to your mask automatically.

## Further reading

Lesion network mapping was introduced by Boes et al. (2015) and has since been
applied across a wide range of neurological and psychiatric conditions. For the
methodological background, see the original functional approach (Boes et al.,
2015; Fox, 2018) and structural disconnection methods (Griffis et al., 2019).
