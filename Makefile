test:
	@clear
	@python app/main.py rag-eval-test --workflow-task-id=123123
	@pytest  --durations=0 -v --cov=app

format:
	@isort app/
	@isort tests/
	@black app/
	@black tests/

to_argo:
	@python scripts/argo-workflow.py add
