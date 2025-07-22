# What Makes a Good TODO Comment?

## dataset

`./data/extracted_introduced_todo` and `./data/extracted_eliminated_todo` are the TODO-introduced dataset and TODO-eliminated dataset, respectively.

`data/todo_data_labeled-f.xlsx` contains the labeled TODO data.

`data/subcategories of task_good TODO.xlsx` and `data/subcategories of notice_good TODO.xlsx` are the subcategories for different forms of TODO.

## code for our classifiers

`./todo-classifier` contains the source code of ml based classifiers and dl based classifiers.

`./todo-classifier/todo_comment_baselines.py` is the script for ml based classifiers.

`./todo-classifier/dl_classifiers.py` is the script for dl based classifiers.

Before running the scripts, please change the model path of CodeBERT to yours path. (highlight with TODO in our scripts)