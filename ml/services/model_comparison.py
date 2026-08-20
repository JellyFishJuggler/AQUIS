def get_comparison(task):
    return {
        "task": task,
        "message": "Model comparison requires trained models. Run /train first.",
        "available_models": [],
        "comparison": {},
    }
