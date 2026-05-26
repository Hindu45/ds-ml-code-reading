# Code Reading Exercises for Data Science and Machine Learning

Data science, like programming in general, can increasingly be streamlined using AI coding support. However:

> Even if you write little code yourself, you'll have to make many decisions in data science. If you work with AI, delegating tasks to it and evaluating what comes out requires strong understanding.

Therefore, understanding code pattern, concepts, and the general data-science process becomes more important than, for example, memorizing exact syntax. Recognizing conceptual and procedural elements in code will remain a valuable skill.

To train this skill, this repo provides ready-made data-science scripts along with code reading questions.

## Contents

- [Repo Setup](#repo-setup)
- [Working With Scripts](#working-with-scripts)
- [Reference Docs](#reference-docs)

## Repo Setup

Run `setup-venv.bat` to create the virtual environment and install all dependencies in one step. It creates `.venv/`, writes the path file for imports, and runs `install-missing-packages.py`. This script checks the base Python installation for existing site-packages and installs only what is missing from `requirements.txt`. This is storage efficient in lab sessions where the Python base distribution already has many data-science packages.

> This repo is read-only. You'll never push changes back. Scripts are meant to be read and run as-is.

---

## Working With Scripts

### Running scripts - interactive mode (cell-by-cell)

Most Python scripts in this repo use interactive cells, marked by `#%%`.

- Open the script in VS Code with the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) extensions.
- Each `#%%` block runs as a Jupyter cell. Best for step-by-step exploration with intermediate output.

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