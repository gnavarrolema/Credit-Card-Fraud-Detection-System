import mlflow
import mlflow.spark
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, count, sum, avg, stddev, max,
    hour, dayofweek, month, year, weekofyear,
    unix_timestamp, current_timestamp, lag, when, round,
    to_timestamp, expr, lit, coalesce
)
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType,
    TimestampType, BooleanType
)
from pyspark.ml import PipelineModel
import logging
from typing import Dict, List
from datetime import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """Configuration for batch inference."""
    NEW_TRANSACTIONS_PATH: str = "/FileStore/fraud_detection/data/new_transactions"
    MODEL_ROOT_PATH: str = "/FileStore/fraud_detection/models/fraud_detector"
    OUTPUT_PATH: str = "/FileStore/fraud_detection/results/scored_transactions"
    FEATURE_METADATA_PATH: str = "/FileStore/fraud_detection/data/features/feature_metadata.json"
    
    # Batch processing configuration
    BATCH_SIZE: int = 10000
    
    # Time windows for feature calculation (in hours)
    TIME_WINDOWS: Dict[str, int] = field(default_factory=lambda: {
        "short": 1,    # 1 hour
        "medium": 24,  # 1 day
        "long": 168    # 1 week
    })
    
    # Schema definition
    SCHEMA: StructType = field(default_factory=lambda: StructType([
        StructField("transaction_id", StringType(), False),
        StructField("customer_id", StringType(), False),
        StructField("timestamp", StringType(), False),
        StructField("amount", FloatType(), False),
        StructField("merchant_category", StringType(), True),
        StructField("merchant_country", StringType(), True),
        StructField("card_present", BooleanType(), True)
    ]))
    
    @classmethod
    def create_paths(cls) -> None:
        """Ensure all necessary directories exist."""
        Path(cls.OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

class BatchInference:
    """Class to handle batch inference operations."""
    
    def __init__(self, spark: SparkSession, config: InferenceConfig):
        self.spark = spark
        self.config = config
        self.model = self._load_best_model()
        self.preprocessor = None  # Initialize as None; will be loaded if it exists
        self.feature_cols = self._load_feature_columns()
        self._setup_spark_configs()
    
    def _setup_spark_configs(self) -> None:
        """Set up optimal Spark configurations for batch processing."""
        self.spark.conf.set("spark.sql.adaptive.enabled", "true")
        self.spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        self.spark.conf.set("spark.sql.shuffle.partitions", "200")
    
    def _load_best_model(self) -> PipelineModel:
        """Load the best model and preprocessor dynamically from the training summary."""
        summary_path = f"{self.config.MODEL_ROOT_PATH}/training_summary.json"
        try:
            if not os.path.exists(summary_path.replace("/FileStore", "/dbfs")):
                raise FileNotFoundError(
                    f"Training summary not found at {summary_path}. "
                    "Please run the training pipeline first."
                )
            with open(summary_path.replace("/FileStore", "/dbfs"), "r") as f:
                summary = json.load(f)
            model_path = summary["best_model"]["path"]
            preprocessor_path = f"{model_path}/preprocessor"
            
            # Load the preprocessor if it exists
            if os.path.exists(preprocessor_path.replace("/FileStore", "/dbfs")):
                logger.info(f"Loading preprocessor from {preprocessor_path}")
                self.preprocessor = PipelineModel.load(preprocessor_path)
            else:
                logger.warning("Preprocessor not found; assuming raw features match training.")
            
            logger.info(f"Loading best model from {model_path}")
            return PipelineModel.load(model_path)
        except Exception as e:
            logger.error(f"Error loading best model: {str(e)}")
            raise
    
    def _load_feature_columns(self) -> List[str]:
        """Load feature columns from the feature metadata."""
        try:
            with open(self.config.FEATURE_METADATA_PATH.replace("/FileStore", "/dbfs"), "r") as f:
                metadata = json.load(f)
            feature_cols = []
            for group_name, group_cols in metadata["feature_groups"].items():
                feature_cols.extend(group_cols)
            logger.info(f"Loaded {len(feature_cols)} feature columns from metadata.")
            return feature_cols
        except Exception as e:
            logger.error(f"Error loading feature metadata: {str(e)}")
            raise
    
    def load_new_transactions(self) -> DataFrame:
        """Load and validate new transactions."""
        try:
            logger.info(f"Loading new transactions from {self.config.NEW_TRANSACTIONS_PATH}")
            
            df = self.spark.read.schema(self.config.SCHEMA).csv(
                self.config.NEW_TRANSACTIONS_PATH,
                header=True,
                mode="DROPMALFORMED"
            ).withColumn(
                "timestamp",
                to_timestamp("timestamp", "yyyy-MM-dd HH:mm:ss.SSSSSS")
            )
            
            # Verify that all required columns are present
            missing_cols = set(self.config.SCHEMA.fieldNames()) - set(df.columns)
            if missing_cols:
                raise ValueError(f"New transactions missing required columns: {missing_cols}")
            
            df = df.filter(
                (col("amount") >= 0) & 
                col("transaction_id").isNotNull() & 
                col("customer_id").isNotNull()
            )
            
            # Check for null values in critical columns
            for column in ["transaction_id", "customer_id", "timestamp", "amount"]:
                null_count = df.filter(col(column).isNull()).count()
                if null_count > 0:
                    logger.warning(f"Found {null_count} null values in {column}")
            
            total_records = df.count()
            logger.info(f"Loaded {total_records} transactions for scoring.")
            return df.repartition(200).cache()
            
        except Exception as e:
            logger.error(f"Error loading new transactions: {str(e)}")
            raise
    
    def generate_features(self, df: DataFrame) -> DataFrame:
        """
        Generate features for model inference, aligned with the feature engineering process.
        """
        try:
            logger.info("Generating features for inference.")
            df.createOrReplaceTempView("new_transactions")
            
            df_with_velocity = self._calculate_velocity_metrics(df)
            df_with_merchant = self._calculate_merchant_profiles(df_with_velocity)
            df_with_time = self._calculate_time_features(df_with_merchant)
            
            # Apply the preprocessor if it exists, then the model
            if self.preprocessor:
                featured_df = self.preprocessor.transform(df_with_time)
                return self.model.transform(featured_df)
            return self.model.transform(df_with_time)
            
        except Exception as e:
            logger.error(f"Error generating features: {str(e)}")
            raise
    
    def _calculate_velocity_metrics(self, df: DataFrame) -> DataFrame:
        """Calculate transaction velocity metrics with optimization."""
        result_df = df
        for window_name, hours in self.config.TIME_WINDOWS.items():
            w = Window.partitionBy("customer_id") \
                      .orderBy(unix_timestamp("timestamp")) \
                      .rangeBetween(-hours * 3600, 0)
            result_df = result_df.withColumns({
                f"tx_count_{window_name}": count("transaction_id").over(w),
                f"amount_sum_{window_name}": sum("amount").over(w),
                f"amount_avg_{window_name}": avg("amount").over(w),
                f"amount_std_{window_name}": coalesce(stddev("amount").over(w), lit(0)),
                f"amount_max_{window_name}": max("amount").over(w)
            })
        return result_df
    
    def _calculate_merchant_profiles(self, df: DataFrame) -> DataFrame:
        """Calculate merchant profile features."""
        merchant_profiles = self.spark.sql("""
            WITH merchant_stats AS (
                SELECT 
                    merchant_category,
                    AVG(amount) as merchant_avg_amount,
                    STDDEV(amount) as merchant_stddev_amount,
                    COUNT(*) as merchant_tx_count,
                    COUNT(DISTINCT customer_id) as merchant_unique_customers
                FROM new_transactions
                GROUP BY merchant_category
            ),
            hour_counts AS (
                SELECT 
                    merchant_category,
                    hour(timestamp) as hour,
                    COUNT(*) as tx_count
                FROM new_transactions
                GROUP BY merchant_category, hour(timestamp)
            ),
            ranked_hours AS (
                SELECT 
                    merchant_category,
                    hour,
                    tx_count,
                    ROW_NUMBER() OVER (PARTITION BY merchant_category ORDER BY tx_count DESC) as rank
                FROM hour_counts
            )
            SELECT 
                m.*,
                COALESCE(r1.hour, -1) as peak_hour_1,
                COALESCE(r2.hour, -1) as peak_hour_2,
                COALESCE(r3.hour, -1) as peak_hour_3
            FROM merchant_stats m
            LEFT JOIN ranked_hours r1 ON m.merchant_category = r1.merchant_category AND r1.rank = 1
            LEFT JOIN ranked_hours r2 ON m.merchant_category = r2.merchant_category AND r2.rank = 2
            LEFT JOIN ranked_hours r3 ON m.merchant_category = r3.merchant_category AND r3.rank = 3
        """)
        return df.join(merchant_profiles, "merchant_category", "left")
    
    def _calculate_time_features(self, df: DataFrame) -> DataFrame:
        """Calculate time-based features."""
        w = Window.partitionBy("customer_id").orderBy("timestamp")
        return df \
            .withColumn("hour_of_day", hour("timestamp")) \
            .withColumn("day_of_week", dayofweek("timestamp")) \
            .withColumn("month", month("timestamp")) \
            .withColumn("year", year("timestamp")) \
            .withColumn("week_of_year", weekofyear("timestamp")) \
            .withColumn("is_weekend", when(dayofweek("timestamp").isin([1, 7]), 1).otherwise(0)) \
            .withColumn("is_night", when((hour("timestamp") >= 23) | (hour("timestamp") < 5), 1).otherwise(0)) \
            .withColumn("time_since_last_tx", 
                       coalesce(round((unix_timestamp("timestamp") - lag(unix_timestamp("timestamp")).over(w)) / 3600, 2), lit(9999)))
    
    def run_predictions(self, df: DataFrame) -> DataFrame:
        """Run model predictions using the loaded model."""
        try:
            logger.info("Generating predictions.")
            predictions = df
            
            scored_df = predictions.select(
                "transaction_id",
                "timestamp",
                "customer_id",
                "merchant_category",
                "amount",
                "probability",
                "prediction"
            ).withColumn("scoring_timestamp", current_timestamp())
            
            return scored_df
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def save_results(self, df: DataFrame) -> None:
        """Save scored transactions with enhanced metadata."""
        try:
            logger.info(f"Saving scored transactions to {self.config.OUTPUT_PATH}")
            
            total_transactions = df.count()
            fraud_predictions_count = df.filter(col("prediction") == 1.0).count()
            avg_probability = df.selectExpr("avg(probability[1]) as avg_prob").first()["avg_prob"] or 0.0
            
            # Add count by merchant category
            category_counts = df.groupBy("merchant_category").count().collect()
            category_counts_dict = {row["merchant_category"]: int(row["count"]) for row in category_counts}
            
            run_metadata = {
                "timestamp": datetime.now().isoformat(),
                "model_path": self.model.stages[-1].uid if self.model.stages else "unknown",
                "total_transactions": total_transactions,
                "scoring_metrics": {
                    "fraud_predictions": fraud_predictions_count,
                    "average_probability": float(avg_probability)
                },
                "category_counts": category_counts_dict
            }
            
            with open(f"{self.config.OUTPUT_PATH}/metadata.json".replace("/FileStore", "/dbfs"), "w") as f:
                json.dump(run_metadata, f, indent=2)
            
            df.write \
                .format("delta") \
                .mode("append") \
                .partitionBy("scoring_timestamp") \
                .save(f"{self.config.OUTPUT_PATH}/predictions")
            
            logger.info("Results saved successfully.")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

def main():
    """Main execution function."""
    try:
        config = InferenceConfig()
        config.create_paths()
        
        spark = SparkSession.builder \
            .appName("FraudDetection-BatchInference") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        inference = BatchInference(spark, config)
        
        transactions_df = inference.load_new_transactions()
        featured_df = inference.generate_features(transactions_df)
        scored_df = inference.run_predictions(featured_df)
        inference.save_results(scored_df)
        
        logger.info("Batch inference pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"Batch inference pipeline failed: {str(e)}")
        raise
    finally:
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    main()
