from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

DATA_PATH = "data/offline.csv"
MODEL_DIR = "models/best_model"

LABEL_COL = "Diabetes_012"


def build_preprocess(feature_cols):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=False, withStd=True)
    return [assembler, scaler]


def train_with_cv(train_df, pipeline, param_grid):
    evaluator = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="f1"
    )

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=2,
        seed=42,
    )

    cv_model = cv.fit(train_df)
    best_model = cv_model.bestModel
    best_f1 = max(cv_model.avgMetrics)

    return best_model, best_f1


def main():
    spark = (
        SparkSession.builder
        .appName("OfflineTraining")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
        .config("spark.hadoop.fs.file.impl.disable.cache", "true")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        .config("spark.sql.warehouse.dir", "file:///D:/rmnp3/spark-warehouse")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.option("header", True).option("inferSchema", True).csv(DATA_PATH)

    df = df.withColumn(LABEL_COL, col(LABEL_COL).cast("int"))

    feature_cols = [c for c in df.columns if c != LABEL_COL]

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    preprocess_stages = build_preprocess(feature_cols)

    lr = LogisticRegression(labelCol=LABEL_COL, featuresCol="features")
    lr_pipe = Pipeline(stages=preprocess_stages + [lr])

    lr_grid = (ParamGridBuilder()
               .addGrid(lr.regParam, [0.0, 0.01, 0.1])
               .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
               .build())

    best_lr, f1_lr = train_with_cv(train_df, lr_pipe, lr_grid)
    print(f"[LR] best CV F1 = {f1_lr:.4f}")

    rf = RandomForestClassifier(labelCol=LABEL_COL, featuresCol="features", seed=42)
    rf_pipe = Pipeline(stages=preprocess_stages + [rf])

    rf_grid = (ParamGridBuilder()
               .addGrid(rf.numTrees, [50, 100])
               .addGrid(rf.maxDepth, [5, 10])
               .build())

    best_rf, f1_rf = train_with_cv(train_df, rf_pipe, rf_grid)
    print(f"[RF] best CV F1 = {f1_rf:.4f}")

    dt = DecisionTreeClassifier(labelCol=LABEL_COL, featuresCol="features", seed=42)
    dt_pipe = Pipeline(stages=preprocess_stages + [dt])

    dt_grid = (ParamGridBuilder()
               .addGrid(dt.maxDepth, [5, 10, 15])
               .addGrid(dt.maxBins, [32, 64])
               .build())

    best_dt, f1_dt = train_with_cv(train_df, dt_pipe, dt_grid)
    print(f"[DT] best CV F1 = {f1_dt:.4f}")

    best_name, best_model, best_f1 = max(
        [("LR", best_lr, f1_lr), ("RF", best_rf, f1_rf), ("DT", best_dt, f1_dt)],
        key=lambda x: x[2]
    )

    print(f"\nBEST MODEL = {best_name} with CV F1 = {best_f1:.4f}")

    evaluator = MulticlassClassificationEvaluator(labelCol=LABEL_COL, metricName="f1")
    test_pred = best_model.transform(test_df)
    test_f1 = evaluator.evaluate(test_pred)
    print(f"Test F1 (offline holdout) = {test_f1:.4f}")

    import shutil, os
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)

    best_model.write().overwrite().save(MODEL_DIR)
    print(f"Saved model to: {MODEL_DIR}")

    spark.stop()


if __name__ == "__main__":
    main()
