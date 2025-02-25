from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, count, sum, avg, stddev, hour, dayofweek, month, year, weekofyear,
    unix_timestamp, current_timestamp, lag, when, round, max, approx_count_distinct,
    array, lit, coalesce
)
import logging
from typing import Dict, List
from datetime import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuración para ingeniería de características."""
    INPUT_PATH: str = "/FileStore/fraud_detection/data/processed"
    OUTPUT_PATH: str = "/FileStore/fraud_detection/data/features"
    
    # Time windows (in hours)
    TIME_WINDOWS: Dict[str, int] = field(default_factory=lambda: {
        "short": 1,    # 1 hour
        "medium": 24,  # 1 day
        "long": 168    # 1 week
    })
    
    # Feature groups
    FEATURE_GROUPS: Dict[str, List[str]] = field(default_factory=lambda: {
        "velocity": [
            "tx_count", "amount_sum", "amount_avg",
            "amount_std", "amount_max"
        ],
        "merchant": [
            "merchant_avg_amount", "merchant_stddev_amount",
            "merchant_tx_count", "merchant_unique_customers",
            "merchant_fraud_rate", "peak_hour_1", "peak_hour_2", "peak_hour_3"
        ],
        "time": [
            "hour_of_day", "day_of_week", "month",
            "year", "week_of_year", "is_weekend",
            "is_night", "time_since_last_tx"
        ]
    })
    
    @classmethod
    def create_paths(cls) -> None:
        """Creates the OUTPUT_PATH directory if it does not exist"""
        Path(cls.OUTPUT_PATH).mkdir(parents=True, exist_ok=True)


class FeatureEngineer:
    """Class for performing feature engineering"""

    def __init__(self, spark: SparkSession, config: FeatureConfig):
        self.spark = spark
        self.config = config
        self._setup_spark_configs()
    
    def _setup_spark_configs(self) -> None:
        self.spark.conf.set("spark.sql.adaptive.enabled", "true")
        self.spark.conf.set("spark.sql.shuffle.partitions", "200")
        self.spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    
    def load_processed_data(self) -> DataFrame:
        """Loads processed data in Delta format"""
        try:
            logger.info(f"Loading processed data from: {self.config.INPUT_PATH}")
            df = self.spark.read.format("delta").load(self.config.INPUT_PATH)
            df.cache()
            record_count = df.count()
            logger.info(f"Loaded {record_count} records successfully")
            required_columns = {"transaction_id", "customer_id", "timestamp", "amount", "merchant_category"}
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            return df
        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            raise
    
    def calculate_velocity_features(self, df: DataFrame) -> DataFrame:
        """Computes speed features for multiple windows"""
        try:
            logger.info("Calculating transaction velocity features")
            result_df = df
            for window_name, hours in self.config.TIME_WINDOWS.items():
                w = Window.partitionBy("customer_id") \
                          .orderBy(unix_timestamp("timestamp")) \
                          .rangeBetween(-hours * 3600, 0)
                for metric in self.config.FEATURE_GROUPS["velocity"]:
                    if metric == "tx_count":
                        result_df = result_df.withColumn(f"{metric}_{window_name}", count("transaction_id").over(w))
                    elif metric == "amount_sum":
                        result_df = result_df.withColumn(f"{metric}_{window_name}", sum("amount").over(w))
                    elif metric == "amount_avg":
                        result_df = result_df.withColumn(f"{metric}_{window_name}", avg("amount").over(w))
                    elif metric == "amount_std":
                        # Rellenar con 0 si stddev es nulo (e.g., menos de 2 transacciones)
                        result_df = result_df.withColumn(f"{metric}_{window_name}", 
                                                        coalesce(stddev("amount").over(w), lit(0)))
                    elif metric == "amount_max":
                        result_df = result_df.withColumn(f"{metric}_{window_name}", max("amount").over(w))
            return result_df
        except Exception as e:
            logger.error(f"Error calculating velocity features: {str(e)}")
            raise

    def create_merchant_profiles(self, df: DataFrame) -> DataFrame:
        """
        Calculates the merchant profile metrics and the three peak hours based on transaction frequency per merchant category
        """
        try:
            logger.info("Creating merchant profile features with peak hours")
            df.createOrReplaceTempView("transactions")
            merchant_profiles = self.spark.sql("""
                WITH merchant_stats AS (
                    SELECT 
                        merchant_category,
                        AVG(amount) as merchant_avg_amount,
                        STDDEV(amount) as merchant_stddev_amount,
                        COUNT(*) as merchant_tx_count,
                        approx_count_distinct(customer_id) as merchant_unique_customers,
                        AVG(CASE WHEN is_fraud = true THEN 1 ELSE 0 END) as merchant_fraud_rate
                    FROM transactions
                    GROUP BY merchant_category
                ),
                hour_counts AS (
                    SELECT 
                        merchant_category,
                        hour(timestamp) as hour,
                        COUNT(*) as tx_count
                    FROM transactions
                    GROUP BY merchant_category, hour(timestamp)
                ),
                ranked_hours AS (
                    SELECT 
                        merchant_category,
                        hour,
                        tx_count,
                        ROW_NUMBER() OVER (PARTITION BY merchant_category ORDER BY tx_count DESC) as rank
                    FROM hour_counts
                ),
                merchant_peak_hours AS (
                    SELECT 
                        merchant_category,
                        MAX(CASE WHEN rank = 1 THEN hour END) as peak_hour_1,
                        MAX(CASE WHEN rank = 2 THEN hour END) as peak_hour_2,
                        MAX(CASE WHEN rank = 3 THEN hour END) as peak_hour_3
                    FROM ranked_hours
                    WHERE rank <= 3
                    GROUP BY merchant_category
                )
                SELECT 
                    m.*,
                    COALESCE(p.peak_hour_1, -1) as peak_hour_1,
                    COALESCE(p.peak_hour_2, -1) as peak_hour_2,
                    COALESCE(p.peak_hour_3, -1) as peak_hour_3
                FROM merchant_stats m
                LEFT JOIN merchant_peak_hours p ON m.merchant_category = p.merchant_category
            """)
            result_df = df.join(merchant_profiles, "merchant_category", "left")
            return result_df
        except Exception as e:
            logger.error(f"Error creating merchant profiles: {str(e)}")
            raise
    
    def create_time_features(self, df: DataFrame) -> DataFrame:
        """Creates features based on time"""
        try:
            logger.info("Creating time-based features")
            w = Window.partitionBy("customer_id").orderBy("timestamp")
            result_df = df.withColumn("hour_of_day", hour("timestamp")) \
                          .withColumn("day_of_week", dayofweek("timestamp")) \
                          .withColumn("month", month("timestamp")) \
                          .withColumn("year", year("timestamp")) \
                          .withColumn("week_of_year", weekofyear("timestamp")) \
                          .withColumn("is_weekend", when(dayofweek("timestamp").isin([1, 7]), 1).otherwise(0)) \
                          .withColumn("is_night", when((hour("timestamp") >= 23) | (hour("timestamp") < 5), 1).otherwise(0)) \
                          .withColumn("time_since_last_tx", 
                                    # Rellenar con 9999 si no hay transacción previa
                                    coalesce(round((unix_timestamp("timestamp") - lag(unix_timestamp("timestamp")).over(w)) / 3600, 2), lit(9999)))
            return result_df
        except Exception as e:
            logger.error(f"Error creating time features: {str(e)}")
            raise
    
    def save_features(self, df: DataFrame) -> None:
        """
        Saves the features in Delta format and updates the metadata with the actual column names
        """
        try:
            logger.info(f"Saving engineered features to: {self.config.OUTPUT_PATH}")
            df.write.format("delta").mode("overwrite") \
              .option("overwriteSchema", "true") \
              .option("optimizeWrite", "true") \
              .save(self.config.OUTPUT_PATH)
            
            velocity_cols = []
            for metric in self.config.FEATURE_GROUPS["velocity"]:
                for window_name in self.config.TIME_WINDOWS.keys():
                    velocity_cols.append(f"{metric}_{window_name}")
            
            updated_feature_groups = {
                "velocity": velocity_cols,
                "merchant": self.config.FEATURE_GROUPS["merchant"],
                "time": self.config.FEATURE_GROUPS["time"]
            }
            
            feature_metadata = {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
                "feature_counts": {
                    "velocity": len(velocity_cols),
                    "merchant": len(self.config.FEATURE_GROUPS["merchant"]),
                    "time": len(self.config.FEATURE_GROUPS["time"])
                },
                "time_windows": self.config.TIME_WINDOWS,
                "total_features": len(df.columns),
                "record_count": df.count(),
                "feature_groups": updated_feature_groups
            }
            
            metadata_path = f"{self.config.OUTPUT_PATH}/feature_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(feature_metadata, f, indent=2)
            
            logger.info("Features and metadata saved successfully")
        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            raise


def main():
    """Main function"""
    try:
        config = FeatureConfig()
        config.create_paths()
        spark = SparkSession.builder \
            .appName("FraudDetection-FeatureEngineering") \
            .getOrCreate()
        engineer = FeatureEngineer(spark, config)
        df = engineer.load_processed_data()
        df = engineer.calculate_velocity_features(df)
        df = engineer.create_merchant_profiles(df)
        df = engineer.create_time_features(df)
        engineer.save_features(df)
        logger.info("Feature engineering pipeline completed successfully")
    except Exception as e:
        logger.error(f"Feature engineering pipeline failed: {str(e)}")
        raise
    finally:
        # spark.stop() si se desea cerrar la sesión
        pass

if __name__ == "__main__":
    main()