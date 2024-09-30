# RAG-Eval
A standard process of RAG/KG evaluation

# how to use
1. git clone
1. install docker
1. install dev container in your vs code
1. ask the project owner privately for a copy of "config"
1. modify the config, can refer to config.template

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

# input
* questions, a list of str
* answers, a list of str
* ground_truths. The true answer, it should be a list of str
* contexts. They are the retrieved chunks, a list of lists of str
* reference. They could be the true chuks to the true answer. They are a list of lists of str