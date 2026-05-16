# Citation Checker

A Python tool that checks whether the references in a scientific paper actually exist — and whether the author names and year are correct. Useful for auditing AI-assisted writing, where hallucinated citations are increasingly common.

Based on the methodology of Zhao et al. (2025), *"LLM hallucinations in the wild: Large-scale evidence from non-existent citations"*.
https://doi.org/10.48550/arXiv.2605.07723

---

## What it does

For each reference you give it, the tool:

1. Parses out the title, authors, year, and DOI
2. Searches Semantic Scholar, CrossRef, and OpenAlex (all free, no account needed)
3. Returns one of three verdicts:

| Verdict | Meaning |
|---|---|
| ✅ **VERIFIED** | Paper found; title, authors, and year all match |
| ⚠️ **SUSPICIOUS** | Paper found, but the year or authors don't match what was claimed |
| ❌ **NOT_FOUND** | No matching paper found in any database — likely hallucinated |

---

## Limitations

- The tool checks whether a paper *exists* with that title, and whether the authors and year *match*. It cannot check whether a real paper actually supports the claim it is cited for — that remains an open research problem.
- Coverage is best for English-language journal articles indexed by Semantic Scholar, CrossRef, or OpenAlex. Theses, book chapters, reports, and non-English sources may not be found even if they are real.
- Roughly 1–2% of real papers may show as NOT_FOUND due to indexing gaps or unusual title formatting.
- For very large .bib files (hundreds of entries), the run time is long due to API rate limits. Consider running overnight or checking a suspicious subset first.

---

## Step 1 — Check if Python is installed

Open a terminal:
- **Windows**: press `Win + R`, type `cmd`, press Enter
- **Mac**: press `Cmd + Space`, type `Terminal`, press Enter
- **Linux**: press `Ctrl + Alt + T`

Then type:

```
python --version
```

If you see something like `Python 3.10.4` you are good. Go to Step 3.

If you see an error like *"python is not recognized"*, try:

```
python3 --version
```

If that also fails, continue to Step 2.

---

## Step 2 — Install Python (if needed)

### Windows

1. Go to https://www.python.org/downloads/
2. Click the yellow **"Download Python 3.x.x"** button
3. Run the installer
4. **Important:** on the first screen, tick the box that says **"Add Python to PATH"** before clicking Install
5. Once installed, close and reopen your terminal, then repeat Step 1

### Mac

The easiest way is via Homebrew. If you don't have Homebrew:

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install Python:

```
brew install python
```

### Linux (Debian/Ubuntu)

```
sudo apt update && sudo apt install python3 python3-pip
```

---

## Step 3 — Download the script

Download `citation_checker.py` and place it somewhere easy to find, for example your Desktop or a folder called `citation-checker`.

If you received it as a file, you already have it. Otherwise save it from wherever it was shared.

---

## Step 4 — Open a terminal in the right folder

You need your terminal to be pointing at the folder where `citation_checker.py` lives.

**Windows:** open File Explorer, navigate to the folder, then in the address bar at the top type `cmd` and press Enter. A terminal opens already in that folder.

**Mac/Linux:** open Terminal, then type `cd` followed by a space and the path to your folder. For example:

```
cd ~/Desktop/citation-checker
```

You can drag the folder from Finder onto the Terminal window to paste its path automatically.

---

## Step 5 — Install the dependencies

The script needs four small libraries. Install them by running:

```
pip install requests rapidfuzz tqdm colorama
```

On some systems (especially Mac/Linux) you may need to use `pip3` instead:

```
pip3 install requests rapidfuzz tqdm colorama
```

You only need to do this once. It should complete in under a minute.

---

## Step 6 — Prepare your references

The tool accepts three file formats and detects them **automatically** from the file extension and content. No conversion needed — use whichever format your reference manager already exports.

### BibTeX (.bib) — recommended if you use LaTeX

Every field is read directly, with no guessing. This gives the most reliable results.

```bibtex
@article{Blanchet1992,
  author  = {L. Blanchet and T. Damour},
  title   = {Hereditary effects in gravitational radiation},
  journal = {Phys. Rev. D},
  volume  = {46},
  pages   = {4304},
  year    = {1992}
}
```

Export from **Zotero**: File → Export Library → format: BibTeX → save as `refs.bib`  
Export from **Mendeley**: File → Export → BibTeX  
Export from **JabRef**: File → Export → BibTeX  
From **LaTeX**: your `.bib` file is already the right format — just use it directly.

### RIS (.ris) — exported by most academic databases

```
TY  - JOUR
AU  - Blanchet, L.
AU  - Damour, T.
TI  - Hereditary effects in gravitational radiation
PY  - 1992
ER  -
```

Export from **Web of Science**, **Scopus**, or **PubMed** using their "Export citations → RIS" option. Zotero and Mendeley can also export RIS.

### Plain text (.txt) — copy-paste from any bibliography

One reference per line, any citation style:

```
[1] S. Farquhar et al., "Detecting hallucinations in large language models using semantic entropy," Nature, 2024.
[2] doi:10.1038/s41586-024-07421-0
```

> **Tip:** want to try the tool before using your own file? Skip this step and use `--mock` in Step 7.

---

## Step 7 — Run the tool

### Try the offline demo first (no files needed, no internet needed)

```
python citation_checker.py --mock
```

This shows you what the output looks like using simulated results — a good first check that everything installed correctly.

### Run on your own .bib file

```
python citation_checker.py myrefs.bib
```

The tool prints the detected format, the number of entries loaded, then checks each one.

### Run on a .ris or .txt file

```
python citation_checker.py myrefs.ris
python citation_checker.py myrefs.txt
```

Exactly the same command — the format is detected automatically.

### Save results to a file

```
python citation_checker.py myrefs.bib --output audit.json
```

Saves the full results (match scores, DOIs, matched author names, all verdicts) to a JSON file you can inspect or process further.

### Live built-in demos (hit real APIs, require internet)

```
python citation_checker.py --demo-bib    # BibTeX demo with 4 entries
python citation_checker.py --demo        # plain-text demo with 7 references
```

Both include real papers, a paper with a wrong year, and a hallucinated citation, so you can see all three verdict types.

---

## Example output

```
Detected format : BIBTEX
Loaded 4 references.
Checking against Semantic Scholar, CrossRef, OpenAlex...

========================================================================
  CITATION CHECKER REPORT
========================================================================

   1. [VERIFIED]  Farquhar2024
      Title   : Detecting hallucinations in large language models using semantic entropy
      Year    : 2024
      Authors : Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn
      Found   : Detecting hallucinations in large language models using semantic entropy  [CrossRef (DOI)]
      Scores  : title=97  authors=82  year=Y
      DOI     : 10.1038/s41586-024-07421-0
      Venue   : Nature

   3. [SUSPICIOUS]  Shumailov2022wrong
      Title   : AI models collapse when trained on recursively generated data
      Year    : 2022
      Found   : AI models collapse when trained on recursively generated data  [Semantic Scholar]
      Scores  : title=98  authors=75  year=N
      DOI     : 10.1038/s41586-024-07566-y
      !  Year mismatch (claimed 2022, found 2024)

   4. [NOT_FOUND]  hallucinated2024
      Title   : Quantum entanglement effects on deep learning convergence rates...
      !  Title not matched in Semantic Scholar, CrossRef, or OpenAlex.

------------------------------------------------------------------------
  SUMMARY  (4 references)
------------------------------------------------------------------------
  VERIFIED         2 ( 50.0%)  #########################
  SUSPICIOUS       1 ( 25.0%)  ############
  NOT_FOUND        1 ( 25.0%)  ############
========================================================================
```

Note that BibTeX results show the **cite key** (e.g. `Farquhar2024`, `Shumailov2022wrong`) instead of a plain number, making it easy to find the entry in your `.bib` file.

---

## Troubleshooting

**Wrong number of references loaded from a .bib file**  
Make sure the file is saved with a `.bib` extension. If it has a `.txt` extension the tool will try to parse it line by line instead of as BibTeX entries.

**`pip` is not recognized**  
Try `pip3` instead of `pip`. If that also fails, try `python -m pip install ...` or `python3 -m pip install ...`.

**`ModuleNotFoundError: No module named 'requests'`**  
The dependencies were not installed, or were installed for a different Python version. Re-run the `pip install` command from Step 5.

**References all show NOT_FOUND**  
Check your internet connection. The tool needs to reach external APIs. If you are on a restricted network (some universities, corporate VPNs), try on a different connection.

**Slow performance**  
The tool makes up to 3 API calls per reference with small delays to avoid rate limiting. For 50 references expect roughly 1–2 minutes; for 368 references, around 15–20 minutes.

**Mac says "python not found" but "python3" works**  
Just replace `python` with `python3` in all commands above. Everything else is identical.

---

## License

This project is licensed under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

**You are free to:**
- Use, copy, and adapt this tool for academic research, teaching, and personal projects
- Share and redistribute it with attribution

**Under the following terms:**
- **Attribution** — cite the original repository and author (D. Banasik) in any publication or derivative work
- **NonCommercial** — commercial use, including integration into paid products or services, requires explicit written permission from the author

For commercial licensing inquiries, open an issue in this repository.
