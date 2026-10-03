FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY sleeve_fund ./sleeve_fund
RUN pip install --no-cache-dir .
COPY configs ./configs
COPY research ./research
RUN useradd --create-home --uid 1000 sleeve && chown -R sleeve /app
USER sleeve
