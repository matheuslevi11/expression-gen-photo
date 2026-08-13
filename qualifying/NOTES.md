# `qualifying/` — working notes

Scaffold for the **Master's Thesis Proposal** (*Proposta de Dissertação de Mestrado*,
PPGI / IC-UFAL): *Nuanced Facial Expression Synthesis in Generative Models*.

Prepared 2026-08-11. This file is for build/setup facts and open decisions.
Research content belongs in [`docs/status.md`](../docs/status.md) and
[`docs/experiments/`](../docs/experiments/) per the repo's
[documentation convention](../CLAUDE.md) — **do not** duplicate results here.

---

## What this folder was, and what changed

It arrived as a complete copy of **someone else's undergraduate thesis** — a
deepfake-detection work ("DSFD", a frozen DINOv3 backbone evaluated on the DF40
benchmark), with only `Identificacao.tex` changed to your name and title. It also
carried an unrelated fragment about a VR table-tennis study in `Cap03`. None of
that content was reusable, so it was removed.

**Removed**

| What | Why |
|---|---|
| `Cap01`–`Cap04` bodies, `Conc/` | Deepfake-detection prose, unrelated to this work |
| `images/` (80 files, 8.4 MB: `df40/`, `families/`, `spectra/`, `aug/`, `domains/`, `train_test/`) | DF40 sample grids and spectra |
| `Ref/SampleReferences.bib` (34 entries) | Deepfake bibliography; 5 general-ML entries were carried over |
| `Cap00/Glossario.tex`, `Simbolos.tex` contents | DF40/DSFD acronyms and BCE notation |
| `Cap00/Resumo.tex`, `Abstract.tex` contents | Abstracts of the other work |
| `svg`, `lipsum`, `pdfpages`, duplicate `booktabs`, unused `tikzset` neuron styles | Unused; `svg` additionally needs `inkscape`, which is not installed here |

> ⚠️ A full copy of the original folder is archived at
> `/tmp/claude-1000/-home-mlevi-Research-expression-gen-photo/597d4516-527e-4111-8265-e9de80e697f2/scratchpad/removed-template-content/qualifying-original`.
> **That is session-scoped scratch space and will not survive indefinitely.** If you
> want a permanent copy, move it somewhere durable now. Nothing was ever committed
> to git, so there is no other undo path.

**Kept:** `JITH.cls` (patched, see below), `Cap00/IC.jpg` (the UFAL IC cover logo,
hard-coded in the class), `Identificacao.tex`, and the folder layout.

---

## Decisions taken

**Language: English.** `JITH.cls` hard-coded Portuguese, so it was patched:

- `babel` switched `brazilian` → `english`.
- Cover, title page and approval page strings translated
  (`Proposta de Dissertação de Mestrado` → `Master's Thesis Proposal`,
  `orientado por` → `advised by`, `Banca Examinadora` → `Examination Committee`,
  `Aprovado em` → `Approved on`, degree names, …).
- Date format changed from `DD de Month de YYYY` to `Month DD, YYYY`.
  **Consequence:** `\setMonth` in `Identificacao.tex` must now be an English month name.
- Institution names left in Portuguese (`Universidade Federal de Alagoas`,
  `Instituto de Computação`) since they are the official names. English
  alternatives are sitting commented out at the top of `JITH.cls` if your advisor
  prefers them.
- Fixed a latent template bug: `docType=5` (doctoral thesis) was unreachable
  because the class tested `=4` twice. Does not affect `docType=2`.

To revert to Portuguese, flip `babel` back and undo those strings — all of them are
inside `\capa`, `\folhaDeRosto` and `\folhaDeAprovacao`.

**Structure: 6 chapters**, the conventional Brazilian proposal shape. Chapter files
were renamed from `CapituloN.tex` to meaningful English names. `Cap00/` keeps its
Portuguese filenames because `JITH.cls` `\input`s them by exact path.

```
Cap01/Introduction.tex        motivation · problem · RQs · objectives · hypotheses · contributions
Cap02/Background.tex          diffusion · conditioning · temporal · GenPhoto · FACS · AU estimation · data · metrics
Cap03/RelatedWork.tex         transfer / editing / generation + positioning table
Cap04/Method.tex              formulation · frozen vs trainable · conditioning · data · eval protocol · baselines
Cap05/PreliminaryResults.tex  ramp · identity · calibration · head-to-head · threats
Cap06/Schedule.tex            completed · remaining · timetable · risks · final remarks
```

Every chapter is a **skeleton**: section headings with labels, `\todo[inline]`
markers where prose goes, and comments naming the source material (usually a dated
file in `docs/experiments/`) so nothing has to be re-derived. `Cap05` has its result
tables **pre-filled** from `docs/status.md` — cross-check each number against its
lab note before submitting, since `status.md` is a living summary and the
`experiments/` entries are the authoritative append-only record.

**Positioning is baked into the skeletons.** The FineFace head-to-head
(`docs/experiments/2026-07-21-fineface-headtohead.md`) found ramp-following to be a
tie, FineFace ahead on absolute calibration, and this work ahead on cross-frame
identity. The comments in `Cap01`, `Cap03` and `Cap05` therefore steer toward
*sequence-native formulation + measurement protocol* as the contribution, and away
from any claim of superior or first-ever AU-conditioned control — FineFace (July
2024) has primacy there. Keep `Cap01` and `Cap06` consistent on this; they get read
together.

---

## Building

```bash
cd qualifying
latexmk -pdf main.tex        # bibtex + makeglossaries handled by .latexmkrc
latexmk -C                   # clean all generated files
```

Explicit sequence, if you prefer it:

```bash
pdflatex main && bibtex main && makeglossaries main && pdflatex main && pdflatex main
```

Verified 2026-08-11: builds clean, 28 pages, no errors, no undefined references or
citations. Build artifacts are gitignored via `qualifying/.gitignore` (including
`main.pdf` — remove that line if you'd rather track it).

**Authoring helpers already wired up**

- `\todo{...}` / `\todo[inline]{...}` — margin and inline notes (`todonotes`).
  They render in the PDF, so an unfinished section is visible rather than silent.
- `\TBD` (in `macros.tex`) — red placeholder for numbers still to be filled, so
  nothing provisional slips into a submitted PDF.
- `easyReview` is loaded with `\setreviewson` for advisor cycles:
  `\add{}`, `\remove{}`, `\replace{old}{new}`, `\comment{text}{note}`,
  `\highlight{}`, `\alert{}`. Switch to `\setreviewsoff` in `main.tex` to render
  clean final text.
- Acronyms: `\gls{key}` / `\glspl{key}`, defined in `Cap00/Glossario.tex`.
  `\glsresetall` at each chapter top re-expands first occurrences. Only acronyms
  actually used are printed, so delete unused entries rather than leaving them.

---

## Open items needing your decision

1. **The date is in the past.** `Identificacao.tex` says `June 29, 2026`, but today
   is 2026-08-11. Set the real defence date before generating a cover.
2. **Third committee member.** `\memberC` is empty (so it is omitted from the
   committee list) while `\filiationC` is filled in. Fill in the name or clear both.
3. **Is a Portuguese `Resumo` required** for an English-language proposal at PPGI?
   Both `Abstract` (EN, first) and `Resumo` (PT) are currently included, which is the
   usual arrangement. If PPGI doesn't require it, delete `Cap00/Resumo.tex` and the
   `\resumo` call in `main.tex`.
4. **Verify the bibliography.** Entries marked `% VERIFY` in `Ref/references.bib`
   were entered from memory. The five related-work entries and the GenPhoto entry
   came from the actual PDFs in `related_work/` and the fork README, so those are
   sound. Everything else needs a check against DBLP or the publisher — venues,
   pages, full author lists.
5. **Keep the refuted hypothesis?** `Cap01`'s H3 (absolute calibration) is already
   refuted by the 2026-07-14 experiment. Reporting a refuted hypothesis honestly is
   the stronger move, but the framing is worth agreeing with your advisor.
6. **Scope of the calibration remedies** (`Cap06`): in scope for the dissertation, or
   stated as a limitation? This is the biggest open scope question, and it drives the
   timetable, since it needs retraining.
7. **`\genexpr` in `macros.tex`** is a placeholder name for the method. Decide what
   the system is called, or drop the macro.
8. **`qualifying/` is untracked in git.** Commit it once you're happy with the
   scaffold, so there's a real undo path from here on.

## Deferred, not forgotten

- `List of Symbols` is scaffolded from the notation `Cap04` is likely to use. If the
  proposal ends up with little formal notation, drop the list and the `\simbolos`
  call rather than padding it.
- `images/` is empty apart from `.gitkeep`. The qualitative figures suggested in
  `Cap05` need frames exported from `inference_output/`.
- `hyperref` emits a benign duplicate-anchor warning because `Abstract` and `Resumo`
  are both unnumbered chapters. Harmless; ignore it.
