# Deploying OutageLens (live demo)

OutageLens is one container: a FastAPI backend that runs the physics and models, serving the static control
room from `web/`. Any host that runs a Dockerfile works. At build time the image trains both GNNs, calibrates
the Bayesian model and runs the default benchmarks (`scripts/warm_cache.py`), so start-up only has to load
them. No GPU needed.

## Option A · Hugging Face Spaces (free, recommended)

The free CPU tier has 2 vCPUs and 16 GB of RAM, so every interaction stays fast.

1. Create a Space with **SDK: Docker** (blank template).
2. Push this folder to the Space repository. Then add this block at the very top of the Space's `README.md`, so
   the Space listens on the app's port:

   ```yaml
   ---
   title: OutageLens
   sdk: docker
   app_port: 8000
   ---
   ```
3. The Space builds the Dockerfile (about 5 minutes) and serves at `https://<user>-outagelens.hf.space`.

Free Spaces pause after about 48 hours without visitors; open the URL the day before and again just before
you present.

## Option B · Render (free tier)

1. Push this folder to a GitHub repository.
2. On [render.com](https://render.com) choose **New → Blueprint** and pick the repository. `render.yaml` does
   the rest: Docker runtime, free plan, health check on `/api/health`.
3. The first build takes about 5 minutes. Your URL looks like `https://outagelens.onrender.com`.

Render's free instance has only 0.1 CPU and sleeps after 15 minutes idle (30–60 s to wake). Because the heavy
work happens at build time, the app stays usable there. Estimated from measured CPU time ×10: start-up ≈ 5 s
plus Python imports, ≈ 0.2 s per slider change, ≈ 16 s for a benchmark re-run, ≈ 21 s for the benchmark at a
non-default sensor coverage. Hugging Face is still the safer choice for the judging slot. **Either way, open
the URL 5 minutes before you present.**

## Option C · Railway / Fly.io / any VPS

```bash
docker build -t outagelens .
docker run -p 8000:8000 outagelens
```

Railway and Fly detect the Dockerfile automatically. They inject `PORT`, which `run.py` honours.

## Local and offline (the backup plan)

The laptop run needs no internet: fonts are bundled and every computation is local.

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/warm_cache.py    # one-time, ~10 s
python run.py                   # prints the URL; picks the next free port if 8000 is taken
```

To let judges open it on their own phones over the venue Wi-Fi, start it on all interfaces and share your
laptop's IP address:

```bash
python run.py --host 0.0.0.0 --port 8000
```

## Demo-day checklist

- [ ] Live URL opened and warmed up 5 minutes before the slot.
- [ ] Laptop copy running locally as a fallback (`python run.py`).
- [ ] Browser zoom at 100–110% on the projector; light room, light theme.
- [ ] Robustness tab visited once, so the benchmark is cached before the judges see it.
