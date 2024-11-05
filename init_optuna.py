import optuna

# Create a study to initialize the database
study = optuna.create_study(storage="sqlite:///db.sqlite3")
# Optionally, you can delete this study if you don't need it
optuna.delete_study(study_name="init_study", storage="sqlite:///db.sqlite3")
