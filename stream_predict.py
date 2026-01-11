import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_json, struct, expr
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler

BOOTSTRAP_SERVERS = "localhost:9092"
IN_TOPIC = "health_data"
OUT_TOPIC = "health_data_predicted"

MODEL_DIR = "models/best_model"
CHECKPOINT_DIR = "C:/spark-tmp/checkpoints/health_predicted"

def build_schema_from_model(model: PipelineModel) -> StructType:
    assembler = None
    for st in model.stages:
        if isinstance(st, VectorAssembler):
            assembler = st
            break
    if assembler is None:
        raise RuntimeError("VectorAssembler stage not found in the loaded PipelineModel.")

    input_cols = assembler.getInputCols()
    schema = StructType([StructField(c, DoubleType(), True) for c in input_cols])
    return schema

def main():
    spark = (
        SparkSession.builder
        .appName("HealthStreamingPredict")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    model = PipelineModel.load(MODEL_DIR)
    schema = build_schema_from_model(model)

    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS)
        .option("subscribe", IN_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = (
        raw.selectExpr("CAST(value AS STRING) AS json_str")
        .select(from_json(col("json_str"), schema).alias("data"))
        .select("data.*")
    )
    for f in schema.fields:
        parsed = parsed.withColumn(f.name, col(f.name).cast("double"))

    pred = model.transform(parsed)

    enriched = pred.withColumn("predicted_class", col("prediction").cast("int"))

    out_df = enriched.select(
        to_json(struct(*[col(c) for c in schema.fieldNames()], col("predicted_class"))).alias("value")
    )

    query = (
        out_df.writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS)
        .option("topic", OUT_TOPIC)
        .option("checkpointLocation", CHECKPOINT_DIR)
        .outputMode("append")
        .start()
    )

    print("Streaming started. Writing predictions to topic:", OUT_TOPIC)
    query.awaitTermination()

if __name__ == "__main__":
    main()
