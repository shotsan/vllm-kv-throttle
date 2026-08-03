FROM vllm/vllm-openai:v0.11.2

LABEL org.opencontainers.image.source="https://github.com/shotsan/vllm-kv-throttle" \
      org.opencontainers.image.description="Reproducible vLLM KV-cache saturation and scheduler throttling proof"

WORKDIR /experiment

COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY download_benchmark.py prepare_prompt.py load_test.py proof_test.py run_server.sh ./
RUN chmod +x run_server.sh \
    && python3 download_benchmark.py \
    && python3 prepare_prompt.py

COPY README.md PROOF.md ./

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=6 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

ENTRYPOINT []
CMD ["./run_server.sh"]
