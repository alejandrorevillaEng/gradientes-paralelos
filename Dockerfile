# Benchmark de CPU reproducible: mismo compilador y mismas opciones en cualquier máquina.
#   docker build -t gradientes .
#   docker run --rm gradientes --experimento escalado
#   docker run --rm -v "$PWD/resultados:/datos" gradientes --csv /datos/mi_cpu.csv
FROM ubuntu:25.10 AS compilacion
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt CMakePresets.json ./
COPY include ./include
COPY src ./src
COPY tests ./tests
COPY cuda ./cuda
COPY python ./python
RUN cmake --preset release && cmake --build --preset release -j && ctest --preset release

FROM ubuntu:25.10
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 python3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home banco
COPY --from=compilacion /src/build/release/gradbench /src/build/release/sistema_demo /usr/local/bin/
COPY --from=compilacion /src/build/release/libgradientes.so /usr/local/lib/
COPY --from=compilacion /src/python/gradientes /usr/local/lib/python3/gradientes
ENV GRADIENTES_LIB=/usr/local/lib/libgradientes.so \
    PYTHONPATH=/usr/local/lib/python3
USER banco
WORKDIR /home/banco
ENTRYPOINT ["gradbench"]
