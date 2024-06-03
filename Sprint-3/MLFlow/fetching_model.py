import mlflow

df = mlflow.search_runs(filter_string="metrics.rmse < 81")
run_id = df.loc[df['metrics.r2'].idxmin()]['run_id']

model = mlflow.sklearn.load_model("mlruns/0/" + run_id + '/artifacts/model')

print(model.get_params())