import requests
import yaml
import uuid



def create_workflow_template(template: dict, namespace: str):
    url = f"https://argo-workflows.platform.dev.motiong.net/api/v1/workflow-templates/{namespace}"
    payload = {
        "resourceKind": "WorkflowTemplate",
        "template": template,
        "namespace": namespace,
        # "submitOptions": {
        #     "labels": f"workflows.argoproj.io/workflow-template=yijun-test-with-dir",
        #     "parameters": ["dir-to-process=pdir", "dir-to-save=sdir"]
        # },
    }
    response = requests.post(url, json=payload)
    print(response.status_code)
    print(response.json())
    return response.json()


# put is impossible to use....
# def put_workflow_template(template: dict, namespace: str):
#     url = f"https://argo-workflows.platform.dev.motiong.net/api/v1/workflow-templates/{namespace}/name-doesnt-matter"
#     payload = {
#         "resourceKind": "WorkflowTemplate",
#         "template": template,
#         "namespace": namespace,
#         # "submitOptions": {
#         #     "labels": f"workflows.argoproj.io/workflow-template=yijun-test-with-dir",
#         #     "parameters": ["dir-to-process=pdir", "dir-to-save=sdir"]
#         # },
#     }
#     payload["template"]["metadata"]["resourceVersion"] = "3"
#     response = requests.put(url, json=payload)
    
#     print(response.status_code)
#     print(response.json())
#     return response.json()


def trigger_workflow_template(template_name: str, workflow_task_id: str ,namespace: str):
    js = {
        "resourceKind": "WorkflowTemplate",
        "namespace": "dev",
        "resourceName": template_name,
        "submitOptions": {
            "labels": f"workflows.argoproj.io/workflow-template={template_name}",
            "parameters": [f"workflow-task-id={workflow_task_id}"]
        },
    }

    # Try this: "submitOptions":{"parameters":["name=value"] }

    response = requests.post(
        f"https://argo-workflows.platform.dev.motiong.net/api/v1/workflows/{namespace}/submit",
        # headers=headers,
        json=js,
    )
    print(response.status_code)


if __name__ == "__main__":
    import sys
    action = sys.argv[1]
    namespace = "dev"
    # if action == "put":
    #     with open("scripts/argo-workflow.yaml") as f:
    #         template = yaml.safe_load(f)
    #         put_workflow_template(template, namespace)
    if action == "add":
        with open("scripts/argo-workflow.yaml") as f:
            template = yaml.safe_load(f)
            create_workflow_template(template, namespace)
    else:
        print(f"unknown action: {action}")