FROM motiongcloud.azurecr.io/motiongcloud/py-3.12-base:0.0.2

ARG DOCKER_HOME="/opt/motiong"
ARG DOCKER_CODE="/opt/motiong/code"
ARG DOCKER_GROUP="motiong"
ARG DOCKER_USER="motiong"
ARG DOCKER_UID=5000


WORKDIR ${DOCKER_CODE}

RUN groupadd -g ${DOCKER_UID} ${DOCKER_GROUP} \
    && useradd -r -u ${DOCKER_UID} -g ${DOCKER_GROUP} -d ${DOCKER_HOME} ${DOCKER_USER} \
    && chown -R ${DOCKER_USER}:${DOCKER_GROUP} ${DOCKER_HOME}

# Install git
RUN mkdir ${DOCKER_HOME}/.ssh && \
    chown -R ${DOCKER_USER} ${DOCKER_HOME}/.ssh && \
    apt-get update && apt-get install -y --fix-missing curl openssh-client git sudo && \
    ssh-keyscan github.com >> ${DOCKER_HOME}/.ssh/known_hosts && \
    echo "alias docker='sudo docker'" > ${DOCKER_HOME}/.bashrc

RUN pip install poetry
COPY poetry.lock pyproject.toml ./

RUN --mount=type=ssh,id=thekey \
    mkdir -p /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts && \
    poetry config virtualenvs.create false && \
    poetry config installer.max-workers 6 && \
    poetry install --without dev --no-interaction && \
    rm poetry.lock pyproject.toml

ENV PATH "$PATH:/opt/motiong/.local/bin"

USER ${DOCKER_USER}

ENV PYTHONPATH=.
# ENV PORT=8000
# ENV HOST=0.0.0.0
# ENV WORKER_NUM=1

COPY --chown=${DOCKER_USER} . .
ENTRYPOINT ["python", "app/main.py"]