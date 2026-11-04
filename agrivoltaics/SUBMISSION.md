# Submission record

- Journal: Artificial Intelligence in Agriculture (Elsevier)
- Title: Light-sharing control for tracker-based agrivoltaics under a yield floor: a simulation study of what learning adds and what shade-response uncertainty costs
- Author: Quang-Vinh Dang, British University Vietnam (ORCID 0000-0002-3877-8024)
- Submitted: 2026-10-19
- Repository state at submission: commit `ad074c9` on `main`

## Files submitted
- Manuscript: `agrivoltaics/pdf/Dang_light-sharing-control-tracker-agrivoltaics_AIA-submission.pdf` (56 pages; source `manuscript/main.tex`)
- Highlights: `agrivoltaics/pdf/Dang_light-sharing-control-tracker-agrivoltaics_highlights.pdf`
- Supplementary material: `agrivoltaics/pdf/Dang_light-sharing-control-tracker-agrivoltaics_supplementary-material.pdf` (8 pages; source `manuscript/supplement.tex`)
- Title, abstract, keywords (plain text): `agrivoltaics/pdf/Dang_light-sharing-control-tracker-agrivoltaics_title-abstract-keywords.txt`

Build: `manuscript/build.sh` compiles the supplement and the main text together (they cross-reference each other).

## Open items
- Deposit the code and results archive (for example on Zenodo) and add the DOI to the data and code availability statement; the text currently says the archive will be deposited at acceptance.
- State the page count (56 plus 8 of supplement) in the cover letter.
- Outstanding minor points from the last internal review round: note in the Table cost caption that the economic analysis uses training seed 0; state the direction of the omitted costs on the net value of agrivoltaics.
- The two non-open-access papers (Marrou et al. 2013, Valle et al. 2017) are noted in `literature/README.md` and are not redistributed.

## Outcome
- Manuscript reference: AIA-D-26-01992.
- Editorial decision, reported 2026-11-04: the manuscript did not pass the initial quality check of the editors and was not sent to external reviewers (desk rejection). No reasons were given in the notice.
- Next step: choose another journal and reformat; the current text, figures and supplement are in this folder.

## Second submission: Agricultural Systems (in preparation)
Changes made for the journal's guide for authors (recommended maximum 8000 words; structured abstract of at most 250 words; 1 to 7 keywords; 3 to 5 highlights of at most 85 characters; graphical abstract required; Option C research-data policy):
- Main text reduced to about 5700 words in 24 to 25 pages (from 56 pages); full methods and results moved to the supplementary material (about 48 pages).
- New section on learning the design margin: conformal calibration and a learned design network (scripts \`src/analysis_conformal.py\`, \`src/learned_design.py\`, \`src/fig_design_learning.py\`).
- Structured abstract (240 words), 7 keywords, 5 highlights (71 to 79 characters), graphical abstract (13 cm x 6.5 cm, 300 dpi).
- Files for upload are in \`agrivoltaics/pdf/\`; sources in \`manuscript/\` (\`build.sh\` compiles the supplement and the main text together).

Open items before submitting:
- Research data: deposit the code and results in a repository (for example Zenodo) and cite the DOI; the journal requires deposit or a statement of why not.
- Declaration of generative AI use: the journal requires a statement before the reference list when such tools were used in preparing the manuscript; decide on the wording (Elsevier policy).
- Read the editorial directions linked from the journal's aims and scope; check the fit once more (whole-farm or landscape-level systems focus).
- The numbers in the design section come from a re-run of the pipeline after the computing environment was rebuilt (weather re-downloaded, caches rebuilt); at pair level they agree with the older design tables of the supplement (70% and LER 1.493 for the mean-response design). The controller and economic results are still those of the original run.
- Round-9 review fixes: pair-level analysis of the design section without overlap between calibration and test pairs, per-crop conformal calibration dropped, learned design compared with the margin frontier with bootstrap intervals, text qualified (single-axis trackers, strict 90% floor, maize).
- Cover letter; suggested reviewers if wanted.
- Compile with the fonts and class used here; check the PDF once more for tables and figure legibility at journal size.

## Second submission: Agricultural Systems — submitted
- Submitted: November 2026 (date of the submission-system confirmation to be added).
- Repository state at submission: commit `f2b9c90` on `main`.
- Files submitted (all in `agrivoltaics/submission_AgriculturalSystems/`): cover letter, abstract, manuscript (LaTeX source zip and PDF, 25 pages), highlights, graphical abstract, supplementary material (49 pages). Declaration of competing interests was completed in the submission system.
- Manuscript reference number: to be added when the confirmation e-mail arrives.
- Still open: the generative-AI declaration (decide and, if applicable, add), the deposit of code and results in a repository with a DOI (the cover letter and data statement say it will be cited before publication), and the date and reference number above.
- Keep the build reproducible: `manuscript/build.sh` compiles the supplement and the main text together; the submission package was compiled standalone from the zip.
