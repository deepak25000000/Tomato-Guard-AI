# Datasets for TomatoGuard AI (links checked while preparing this project)

Check each licence on the ORIGINAL source before you publish a paper or ship the app. "Seen" = what the page showed.

| # | Dataset | Role | Where | Licence (seen) |
|---|---|---|---|---|
| 1 | Mendeley 7-class tomato leaf set (yours) | lab baseline | your existing download | your paper's reference |
| 2 | **PlantDoc** (cropped, classification) | field domain-shift test / light fine-tune | https://github.com/pratikkayal/PlantDoc-Dataset  (paper: https://arxiv.org/pdf/1911.10317) | CC BY 4.0 (repo README) |
| 3 | **FieldPlant** (tomato, corn, cassava; Cameroon) | field data, real plantations | paper: https://doaj.org/article/db83acec9d364a93a2e264d7ef465f70 ; Roboflow copy: https://universe.roboflow.com/plant-disease-detection/fieldplant | CC BY 4.0 on Roboflow pages; confirm on the paper's own data link |
| 4 | **CCMT** (Ghana; tomato = 5,435 raw images) | field-style tomato | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10285554/  (Data in Brief 2023, DOI 10.1016/j.dib.2023.109306 - data link is inside the article) | see article |
| 5 | **TOM2024** (tomato, onion, maize; 14 tomato classes incl. virosis, Tuta absoluta) | more classes, pests | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11871477/  (DOI 10.1016/j.dib.2025.111357) | see article |
| 6 | **Tomato Leaf and Fruit Disease Image Dataset** (3,500 field images, 2026) | field images, leaf curl / bushy stunt | https://data.mendeley.com/datasets/swtt5wy2pr/1 | see Mendeley page |
| 7 | **TOBRFV-LMID** (tomato + pepper, RGB / VNIR / 800 nm / 1000 nm) | ToBRFV + spectral study (Phase 3 only) | article: https://doi.org/10.1016/j.dib.2026.112927 - Zenodo link is in the article's data-availability section | CC BY 4.0 (authors' announcement) |
| 8 | Your own phone photos | the only phone-camera domain-shift test | export from the app (History -> confirmed labels) | yours |

Not verified / not included: a 242-image ToBRFV RGB set from an IEEE 2025 paper (could not confirm public download);
PlantVillage (no link checked - the Mendeley lab set already fills the lab role).

## Practical notes
* **Folder layout:** the harmoniser needs images inside class-named folders (class = the image's parent folder).
  Roboflow copies are usually detection format - export "folder structure" classification, or use the original release.
* **PlantDoc:** a cleaned re-release of the detection version reports byte-identical images in both original train and
  test splits (https://huggingface.co/datasets/KeenForgeAI/PlantDoc-corrected). Never trust a dataset's own split - the
  harmoniser removes exact duplicates and groups near-duplicates across ALL datasets.
* **Label mapping** (datasets/default_rules.json, first match wins):
  Bacterial Spot / Early Blight / Late Blight / Healthy / Yellow Leaf Curl Virus are the five core classes;
  everything else that is a leaf problem (Septoria, mold, mosaic, mites, leaf miner, magnesium, wilt ...) -> Other Problem;
  non-tomato crops -> Not Tomato (set "not_tomato_unless": "tomato" for PlantDoc and FieldPlant).
  **Skipped on purpose:** fruit-only classes, "leaf curl(ing)" folders (ambiguous), FieldPlant "Mildiou" - verify against
  the dataset paper, then edit the rules file if you decide to map them. Review `unified/skipped.csv` after every run.
* TOBRFV-LMID is NOT part of the 7-label RGB leaf model - it feeds the separate spectral/fusion study.
