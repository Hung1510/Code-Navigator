# A VHS recorder that actually has CodeNavigator inside it.
#
# The stock ghcr.io/charmbracelet/vhs image ships vhs + ttyd + ffmpeg and
# nothing else — no Python, so `Require codenav` in the tape fails instantly.
# This adds the runtime.
#
# Build:  docker build -t codenav-vhs -f demo/Dockerfile .
# See demo/README.md for the full recording procedure.

FROM ghcr.io/charmbracelet/vhs

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv git \
    && rm -rf /var/lib/apt/lists/*

# A venv keeps pip off the system Python (Debian marks it externally-managed)
# and puts `codenav` on PATH without --break-system-packages games.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# fastembed caches BGE here. Mount a named volume at /root/.cache so the
# ~130MB download survives between runs — otherwise every render re-downloads
# it, and you'd be recording a progress bar instead of a demo.
ENV FASTEMBED_CACHE_PATH=/root/.cache/fastembed

WORKDIR /vhs
