FROM python:3.12-slim
WORKDIR /app
# uv (pinned version) is the only install path: `uv sync --frozen` installs
# exactly the dependency set in uv.lock and FAILS THE BUILD if pyproject.toml
# and uv.lock disagree. pip install . would re-resolve open-ended ranges.
COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /usr/local/bin/uv
# git is required at runtime: audits shell out to git for pull, SHA, and
# check-ignore.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable
# Run as non-root; no shell tools needed beyond what's installed.
RUN useradd -m oi && mkdir -p /data /home/oi/.config/oi \
    && chown -R oi /app /data /home/oi/.config/oi
USER oi
ENV PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"
# Container paths: config at /home/oi/.config/oi/config.toml, secrets via
# env_file, state db on the /data volume (override with db_path in config).
ENTRYPOINT ["oi"]
CMD ["run"]
