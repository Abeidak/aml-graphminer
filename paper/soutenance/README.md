# Soutenance (defence slides)

`AML-GraphMiner_soutenance.pptx` — 16 colourful, animated slides in French: large figures,
native PowerPoint charts, icons and motif diagrams, very little text on screen. **The talk is in
the speaker notes** of each slide (PowerPoint: *View → Notes*, or presenter view). Elements
appear one group after another when a slide opens, and slides change with a fade.

Every number comes from the result files — nothing is typed by hand:

```bash
python paper/soutenance/make_data.py            # measured numbers -> deck_data.json
cd paper/soutenance
npm install                                     # pptxgenjs, react-icons, sharp (once)
node build_deck.js                              # -> AML-GraphMiner_soutenance.pptx
python animate.py AML-GraphMiner_soutenance.pptx  # transitions + entrance animations
```

Sources: `benchmarks/results_parts/*/*/results.csv` and `per_pattern_recall.csv`,
`benchmarks/results_parts/hybrid/hybrid.json`, `results/eda-*/summary.json`, plus the few
figures quoted with their source in `make_data.py` (J2 history, confidence interval, UNODC, tests).
Before the defence: rehearse with the notes (slide 1 carries the two authors and the supervisor).
