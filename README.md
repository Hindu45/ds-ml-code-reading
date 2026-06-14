# Data Science: Code and Output Reading

Data science, like programming in general, can increasingly be streamlined using AI coding support. However, even if code comes from somewhere else, you'll need to be able to make critical decisions in data science. What question to solve? What models to use? Is the pipeline right? What does this output mean for our next steps?

Understanding code pattern, concepts, and the general data-science process is nowadays more important than, for example, memorizing exact syntax. Recognizing conceptual and procedural elements in code as well as being able to reason on results will remain a valuable skill.

To train this skill, this repo provides ready-made data-science scripts along with code and output reading questions.

> **Important note:** This repo is read-only. You'll never push changes back. Scripts are meant to be read and run as-is.

## Contents

- [Repo Content](#repo-content)
- [Repo Setup](#repo-setup)
- [Working With Scripts](#working-with-scripts)
- [Reference Docs](#reference-docs)

## Repo Content

The `cases/` folder contains subfolders for a variety of datasets. In each of these folders, you'll find
- a `README.md`: domain context, code book for the dataset
- a `QUESTIONS.md`: This files contains reading questions for each of the script - follow these
- a numbered sequence of scripts, e.g. `tips_01_eda.py`, `tips_02_single_feature.py`, ... (dataset folder usually start with an exploratory data analysis / EDA script)

A canonical reading sequence is given in [Reading Sequence Index](docs/reading-index.md).

## Repo Setup

This course is working with VS Code as the primary IDE.

**Install VS Code Extensions:** In VS Code, make sure you have the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) extensions installed.

**Setup Python environment:**
- **Default (lab) setup:** On Windows, run `setup-venv.bat` to create a virtual environment and install all dependencies in one step (double-click from file explorer or run `.\setup-venv.bat` from terminal). What the script does:
    - The script creates `.venv/`, writes the path file for imports, and runs `install-missing-packages.py`.
    - Then, the script checks the base Python installation for existing site-packages and installs only what is missing from `requirements.txt`. This is storage efficient in lab sessions where the Python base distribution already has many data-science packages.
- **Private laptops:** If you work on a private laptop, you may need to accomodate for you specific machine, particularly make sure you have an appropriate base Python installation. If you use `setup-venv.bat`, you may need to switch the Python path inside.

---

## Working With Scripts

### Running scripts - interactive mode (cell-by-cell)

Most Python scripts in this repo use interactive cells, marked by `#%%`.

- Open the script in VS Code: start with `cases/seaborn_tips/tips_01_eda.py` script.
- Run `#%%` blocks individually as an interactive cell. You can alway do that from within a cell: (keyboard shortcut: `Ctrl+Enter`). By running individual cells, you can explore intermediate output step-by-step.
- **Check:** In the interactive shell window that opens, make sure that the correct Python interpreter is selected (should be showing the one from your virtual environment)%

See also [interactive mode guide](docs/interactive-mode-guide.md) for detailed setup and troubleshooting.

### Scratch-editing scripts

Although this repo is read-only, you'll sometimes want to scratch-edit scripts. This may be useful for code understanding. **To scratch-edit a script**, copy it first and add `_scratch` to the filename (or whatever you prefer). Or move it to your own `my-work/` folder etc.

Edit and run your copies freely. The original stays untouched, so pulling updates always works cleanly.

**If you accidentally edited an original file** and git/GitHub Desktop shows unexpected changes, discard them before pulling:

1. Open **GitHub Desktop**
2. In the **Changes** tab, you'll see the modified file(s)
3. Right-click the file → **Discard Changes** (restores the original)
4. Then pull: **Repository** menu → **Pull**

If you want to keep some of those changes, move them to copies as suggested above.

---

## Reference Docs

- [Learning Design](docs/learning-design.md): how labs are structured, what you'll practice, and AI policy (didactics)
- [Question Catalogue](docs/question-catalogue.md): question types organized by CRISP-DM phases
- [Reading Sequence Index](docs/reading-index.md): suggested progression for the code reading.

---

As always: Happy learning, happy life! 🫶