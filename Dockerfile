FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY sleeve_fund ./sleeve_fund
RUN pip install --no-cache-dir .
COPY configs ./configs
COPY research ./research
# The writable volumes' mount points, owned by sleeve: Docker gives an empty new volume the owner of
# the folder it mounts on, so a fresh volume is writable (volume-init in compose fixes older ones).
RUN useradd --create-home --uid 1000 sleeve && chown -R sleeve /app \
    && mkdir -p /data/history /data/research && chown sleeve /data/history /data/research
USER sleeve
