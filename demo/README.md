# Recording the demo GIF

`docs/demo.gif` is **generated from `demo.tape`**, not screen-recorded. That
matters: when the CLI's output changes, you re-run one command and the README
is honest again. A hand-recorded GIF rots silently.

## The two things that will bite you

1. **The stock `vhs` Docker image has no Python.** It ships `vhs` + `ttyd` +
   `ffmpeg` and nothing else, so `Require codenav` fails immediately. Hence
   `demo/Dockerfile`.
2. **The model cache must be warm before you record.** `codenav index`
   downloads BGE-small (~130MB) on first use. If that happens on camera, your
   hero GIF opens with a two-minute progress bar. Warm it once, then record.

---

## Route A — WSL2 (simplest if you already have it)

```bash
# One-time deps
sudo apt update && sudo apt install -y ffmpeg
sudo wget -O /usr/local/bin/ttyd \
  https://github.com/tsl0922/ttyd/releases/latest/download/ttyd.x86_64
sudo chmod +x /usr/local/bin/ttyd
sudo snap install vhs        # or: go install github.com/charmbracelet/vhs@latest

# One-time project setup
cd ~/CodeNavigator           # see the note on filesystems below
pip install -e ".[treesitter]"

# WARM THE CACHE — do this before recording, not during
codenav index .

# Record
vhs demo/demo.tape
```

**Filesystem note:** don't record from `/mnt/d/...`. WSL's bridge to the
Windows filesystem is slow enough to visibly drag out the index step in the
recording. `cp -r` the repo into the Linux filesystem (`~/CodeNavigator`) and
record there.

---

## Route B — Docker (no WSL setup)

```powershell
docker build -t codenav-vhs -f demo/Dockerfile .

# First run: installs the package and downloads the model into a named volume.
# Slow. This is the warm-up, and it is NOT the take you keep.
docker run --rm -v ${PWD}:/vhs -v codenav-cache:/root/.cache codenav-vhs `
  bash -c "pip install -e '.[treesitter]' && codenav index ."

# Second run: everything cached. This is the take.
docker run --rm -v ${PWD}:/vhs -v codenav-cache:/root/.cache codenav-vhs `
  bash -c "pip install -e '.[treesitter]' && vhs demo/demo.tape"
```

The `codenav-cache` volume persists the model between runs. Drop it
(`docker volume rm codenav-cache`) only if you want to re-test a cold start.

---

## Iterating

Expect **3–4 render cycles** to get the pacing right. That's not failure,
that's the point — each cycle is one command, not a re-recording.

Watch for:

- **Legibility of the `impact` and `tests` frames.** These are the reason the
  GIF exists. If a viewer can't read them, nothing else matters. Everything
  else in the tape can be rough.
- **Total length.** Target 30–40s. Past that, people scroll away.
- **Dead air.** If a `Sleep` leaves you staring at an idle prompt, cut it.

## File size

GitHub proxies README images through camo, and a heavy GIF loads visibly
slowly — which defeats the purpose. **Target under 5MB.**

```bash
ls -lh docs/demo.gif
```

If it's over, in this order:

1. `Set Framerate 24` — already set (the default 50 roughly doubles the size,
   and terminal output doesn't need it).
2. `Set Width 1100` / `Set FontSize 15` — shrink the canvas.
3. **Cut the `search` frame.** It's the least differentiated part of the demo
   — every RAG tool can do cited search. `impact` and `tests` are the reason
   anyone stars this repo. Sacrifice the generic part first.
