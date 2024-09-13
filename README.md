# RAG-Eval
A standard process of RAG/KG evaluation

# how to setup your repo
* go to github, create repo
* git clone
* code xxx
* now you are in vscode
* get a powershell
* run some commands

    ```
    cp -R ../generic-pdf-processor/.devcontainer .
    cp ../generic-pdf-processor/Makefile .
    ```

* replace the name in the .devcontainer/devcontainer.json, 
* check docker-compose.yaml to make sure you have all you need, change the port if needed
* reopen in container
* a few commands 

    ```
    mkdir app tests scripts
    touch app/main.py
    mkdir app/services app/utils
    mkdir volume_for_mount
    mkdir volume_for_mount/inputs volume_for_mount/outputs volume_for_mount/temp
    touch volume_for_mount/inputs/.keep volume_for_mount/outputs/.keep volume_for_mount/temp/.keep
    poetry init
    - Name set as "app"
    _ The rest just skip
    # blabla
    poetry add -G dev black isort pytest pytest-cov pyyaml 
    poetry add pydantic-settings requests

    poetry source add --priority=supplemental buf https://buf.build/gen/python
    poetry add git+ssh://git@github.com:motiong-io/workflow-task-messenger.git
    poetry add git+ssh://git@github.com:motiong-io/python-file-manager.git

# how to install llama-index

* poetry add llama-index-core
* poetry add llama-index-embeddings-openai

# how to install ragas

* poetry add ragas

# enable Motion G API

    ```bash
    poetry source add --priority=supplemental buf https://buf.build/gen/python
    poetry add motiong-io-motiongapis-grpc-python
    # the update seems no use, so i just remove then add...
    poetry update motiong-io-motiongapis-grpc-python
    poetry remove motiong-io-motiongapis-grpc-python
    ```