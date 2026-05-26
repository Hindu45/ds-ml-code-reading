# Working with Interactive Mode in VS Code

- [Getting started](#getting-started)
- [Running scripts](#running-scripts)
- [Restarting the kernel](#restarting-the-kernel)
- [Troubleshooting](#troubleshooting)

## Getting started

Requires the [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) VS Code extension. Install it once from the Extensions panel (`Ctrl+Shift+X`) if not already present.

Before opening any script, you may warm up the kernel (aka: make sure you have an alive kernel):

1. Open `cases/warmup.py`
2. Click the run icon → **Run in Interactive Window**
3. Wait for the message: `Happy learning, happy life!` (or whatever the random number generator picks)

The kernel is now live. Subsequent scripts will start more reliably.

## Running scripts

- Open a script (e.g. `penguins_01_eda.py`)
- Click the run icon (top right) → **Run in Interactive Window**
- Scripts are numbered — usually run them in order (01 → 02 → 03 ...)

## Restarting the kernel

Restart when:
- You edited a shared module and imports feel stale (for pure code-reading rarely the case)
- Something behaves unexpectedly and you want a clean state

How: click the **Restart** button in the Interactive Window toolbar, or open the Command Palette (`Ctrl+Shift+P`) → `Jupyter: Restart Kernel`.

---

## Troubleshooting

**Kernel hangs on startup**
Wait up to 30 seconds — Windows startup is slow. If nothing happens, kill the process (see below) / close interactive shell / re-open VS Code. Then run `warmup.py` again.

**"Run in Interactive Window" does nothing**
The kernel is still starting. Wait before clicking again. Multiple clicks queue up and make things worse.

**Kernel hangs on restart**
The old process may not have exited. Kill it (see below), then restart from the toolbar.

**Stale imports after editing a shared module**
Restart the kernel and re-run scripts in order from 01.

**Interactive window does not appear**
Open the Command Palette (`Ctrl+Shift+P`) → `Jupyter: Create Interactive Window`.

**How to kill a hung Python process**

- Task Manager: `Ctrl+Shift+Esc` → find `python.exe` → End Task.

- PowerShell terminal: `Stop-Process -Name python -Force` (ok, probably not working without admin)
