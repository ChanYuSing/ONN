# Technical Paper Draft — Working Folder

This folder contains the section-by-section draft of the CAES9541 / EEE technical paper, condensed from the final SDP report in `../report/`.

## Build order (paste into Word template in this order)

1. `00_title_abstract.md` — Title, author block, Abstract, Index Terms (one column)
2. *(Insert continuous section break, switch to two columns from here)*
3. `01_introduction.md` — Section I
4. `02_related_work.md` — Section II
5. `03_methodology.md` — Section III
6. `04_results_discussion.md` — Section IV (Results & Discussion combined)
7. `05_conclusion.md` — Section V
8. `06_acknowledgement_references.md` — Acknowledgement + References

## Target

- 6–12 pages, A4 portrait, two-column body, 12 pt Times New Roman
- ~8 figures/tables max (re-use those already rendered in `../report/figures/`)
- Equations numbered (1)–(11), Word-LaTeX compatible
- Index Terms (alphabetical): Angular Spectrum Method, Diffractive Neural Network, Liquid-Crystal Display, MNIST Classification, Optical Computing

## Source-to-paper mapping

| Paper section | Condensed from |
|---|---|
| Abstract | `report/01_front_matter.md` (rewritten to ~250 w) |
| I. Introduction | `report/02_introduction.md` §1.1, §1.2 |
| II. Related Work | `report/03_theoretical_principles.md` (compressed) |
| III. Methodology | `report/04_system_design.md` §3.1–§3.7 (selected) |
| IV. Results & Discussion | `report/05_implementation_progress.md` + `report/06_challenges_solutions.md` |
| V. Conclusion | `report/08_conclusion.md` |
| References | `report/08_references.md` (verbatim, IEEE) |
