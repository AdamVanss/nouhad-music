# TouchDesigner + Music_model (Python prepares, TD displays)

## 1. Export assets (on your PC)

After `src.separate` (or oracle stems), run:

```powershell
cd c:\Users\lenovo\Documents\Music_model
.\.venv\Scripts\Activate.ps1
python scripts/export_td_assets.py --folder outputs\separated\tame_28_50
```

Outputs in `outputs/td/`:

| File | Use in TD |
|------|-----------|
| `energy.json` | Animated bar heights / scale per stem (30 FPS RMS) |
| `spectrograms/*.png` | Textures (one per stem) |
| `manifest.json` | Full paths for your machine |

Re-run export whenever you generate new WAVs.

## 2. TD network (minimal)

### Spectrograms (static textures)

1. **Movie File In TOP** → set file to `outputs/td/spectrograms/vocals.png` (use full Windows path).
2. Duplicate for drums / bass / other; layout with **Transform TOP** or **Grid** in a **Composite**.
3. Optional: **Switch TOP** + buttons to show one stem at a time for the demo.

### Energy curves (animated)

1. **Text DAT** `energy_loader` — paste path once (edit `ENERGY_PATH`):

```python
ENERGY_PATH = r'C:\Users\lenovo\Documents\Music_model\outputs\td\energy.json'
```

2. **Timer CHOP** or **Timeline** — drive frame index while audio plays.

3. **Script CHOP** `energy_script` — channel per stem:

```python
import json

def onCook(scriptOp):
    scriptOp.clear()
    path = op('energy_loader').module.ENERGY_PATH
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    fps = data['fps']
    t = op('timer1')['timer_fraction'].eval() if op('timer1') else 0.0
    idx = int(t * data['frame_count']) % data['frame_count']
    for name, curve in data['stems'].items():
        scriptOp.appendChan(name)
        scriptOp[name][0] = curve[idx]
    return
```

(Replace `timer1` with your clock: **Audio Play** sync, **Timer CHOP**, or `absTime.frame`.)

4. **Transform SOP** / **Constant CHOP** — scale geometry with `op('energy_script')['vocals']`, etc.

### Audio playback

**Audio File In CHOP** → same clip you exported → **Audio Device Out CHOP**.  
Align timer length with `duration_seconds` in `energy.json`.

## 3. Club talking points

- **Python / PyTorch**: train separator, write WAVs.
- **export_td_assets.py**: turn WAVs into **JSON + images** TD can read.
- **TouchDesigner**: real-time layout; no training inside TD.

## 4. Tips

- Use **absolute paths** on Windows in Movie File In / JSON.
- After re-export, pulse **reload** on Movie File In or restart cook.
- TD Python is not your `.venv`; only `json` + file I/O is needed for this flow.
