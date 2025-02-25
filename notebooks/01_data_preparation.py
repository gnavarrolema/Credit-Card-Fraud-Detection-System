from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, to_timestamp, count, when, isnan, lit,
    current_timestamp, expr
)
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType,
    TimestampType, BooleanType
)
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import json
from pathlib import Path

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class Config:
    """Configuration class for paths and settings"""
    INPUT_PATH = "/FileStore/fraud_detection/data/credit_card_transactions.csv"
    OUTPUT_PATH = "/FileStore/fraud_detection/data/processed"
    SCHEMA = StructType([
        StructField("transaction_id", StringType(), False),
        StructField("customer_id", StringType(), False),
        StructField("timestamp", StringType(), False),
        StructField("amount", FloatType(), False),
        StructField("merchant_category", StringType(), True),
        StructField("merchant_country", StringType(), True),
        StructField("card_present", BooleanType(), True),
        StructField("is_fraud", BooleanType(), False)
    ])
    
    # Validation thresholds
    MAX_NULL_PERCENTAGE = 0.1  # Max allowed percentage of nulls
    MIN_TRANSACTIONS = 1000    # Minimum expected transactions
    
    @classmethod
    def create_paths(cls) -> None:
        """Ensure all necessary directories exist"""
        # The entire OUTPUT_PATH directory is created
        Path(cls.OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

def load_transaction_data(spark: SparkSession) -> DataFrame:
    """
    Load credit card transaction data from CSV file with enhanced error handling
    and validation.
    
    Args:
        spark: Active SparkSession
        
    Returns:
        DataFrame: Raw transaction data
        
    Raises:
        DataValidationError: If data doesn't meet basic quality requirements
    """
    try:
        logger.info(f"Loading data from: {Config.INPUT_PATH}")
        
        # Read CSV with defined schema and options
        df = spark.read.options(
            header=True,
            mode="DROPMALFORMED",
            timestampFormat="yyyy-MM-dd HH:mm:ss"
        ).schema(Config.SCHEMA).csv(Config.INPUT_PATH)
        
        # Perform basic validation
        total_records = df.cache().count()
        
        if total_records < Config.MIN_TRANSACTIONS:
            raise DataValidationError(
                f"Insufficient data: {total_records} records found, "
                f"minimum {Config.MIN_TRANSACTIONS} required"
            )
        
        logger.info(f"Successfully loaded {total_records} records")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def validate_data(df: DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Perform comprehensive data validation
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing validation metrics and statistics
        
    Raises:
        DataValidationError: If data quality thresholds are exceeded
    """
    try:
        logger.info("Performing data validation")
        
        validation_metrics = {
            "null_metrics": {},
            "quality_metrics": {},
            "distribution_metrics": {}
        }
        
        # Check for null values with percentage
        for column in df.columns:
            if column == "amount":
                # For 'amount', also check for NaN
                null_count = df.filter(
                    (col(column).isNull()) |
                    (col(column) == "") |
                    (isnan(col(column)))
                ).count()
            else:
                # For other columns, just check null or empty string
                null_count = df.filter(
                    (col(column).isNull()) |
                    (col(column) == "")
                ).count()
            
            null_percentage = null_count / df.count()
            validation_metrics["null_metrics"][column] = {
                "null_count": null_count,
                "null_percentage": null_percentage
            }
            
            if null_percentage > Config.MAX_NULL_PERCENTAGE:
                raise DataValidationError(
                    f"Column {column} has {null_percentage:.2%} null values, "
                    f"exceeding threshold of {Config.MAX_NULL_PERCENTAGE:.2%}"
                )
        
        # Check for duplicate transactions
        duplicates = df.groupBy("transaction_id").count().filter(col("count") > 1)
        duplicate_count = duplicates.count()
        validation_metrics["quality_metrics"]["duplicate_transactions"] = duplicate_count
        
        # Validate amount ranges
        amount_metrics = df.select(
            count(when(col("amount") <= 0, True)).alias("invalid_amounts"),
            expr("percentile_approx(amount, array(0.05, 0.95))").alias("amount_percentiles")
        ).collect()[0]
        
        validation_metrics["distribution_metrics"]["amount"] = {
            "invalid_count": amount_metrics["invalid_amounts"],
            "percentile_05": float(amount_metrics["amount_percentiles"][0]),
            "percentile_95": float(amount_metrics["amount_percentiles"][1])
        }
        
        logger.info("Data validation completed successfully")
        return validation_metrics
    
    except Exception as e:
        logger.error(f"Error in data validation: {str(e)}")
        raise

def clean_data(df: DataFrame, validation_metrics: Dict) -> Tuple[DataFrame, Dict]:
    """
    Clean and prepare the transaction data
    
    Args:
        df: Raw DataFrame
        validation_metrics: Dictionary of validation metrics
        
    Returns:
        DataFrame: Cleaned data with processing metadata
    """
    try:
        logger.info("Starting data cleaning process")
        cleaning_metrics = {}
        
        # Convert timestamp to proper format
        df = df.withColumn(
            "timestamp",
            to_timestamp("timestamp")
        )
        
        # Remove any duplicate transactions
        initial_count = df.count()
        df = df.dropDuplicates(["transaction_id"])
        cleaning_metrics["duplicates_removed"] = initial_count - df.count()
        
        # Fill missing values with custom logic
        df = df.na.fill({
            "merchant_category": "UNKNOWN",
            "merchant_country": "UNKNOWN",
            "card_present": False
        })
        
        # Remove transactions with invalid amounts
        df = df.filter(col("amount") > 0)
        
        # Add processing metadata
        df = df.withColumn(
            "processing_timestamp",
            current_timestamp()
        ).withColumn(
            "data_version",
            lit("1.0")
        )
        
        # Calculate cleaning metrics
        final_count = df.count()
        cleaning_metrics["total_removed"] = initial_count - final_count
        cleaning_metrics["final_count"] = final_count
        
        logger.info(
            f"Data cleaning completed. Removed {cleaning_metrics['total_removed']} "
            f"invalid records. Final count: {cleaning_metrics['final_count']}"
        )
        
        return df, cleaning_metrics
    
    except Exception as e:
        logger.error(f"Error in data cleaning: {str(e)}")
        raise

def save_processed_data(
    df: DataFrame,
    validation_metrics: Dict,
    cleaning_metrics: Dict
) -> None:
    """
    Save the processed data and metadata in Delta format
    
    Args:
        df: Processed DataFrame
        validation_metrics: Dictionary of validation metrics
        cleaning_metrics: Dictionary of cleaning metrics
    """
    try:
        logger.info(f"Saving processed data to: {Config.OUTPUT_PATH}")
        
        # Save as Delta table with optimizations
        df.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .option("mergeSchema", "true") \
            .save(Config.OUTPUT_PATH)
        
        # Prepare and save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
            "data_metrics": {
                "initial_count": validation_metrics["quality_metrics"].get("total_records", 0),
                "final_count": cleaning_metrics["final_count"],
                "removed_records": cleaning_metrics["total_removed"]
            },
            "validation_metrics": validation_metrics,
            "cleaning_metrics": cleaning_metrics
        }
        
        metadata_path = f"{Config.OUTPUT_PATH}/metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Data and metadata saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise

def main():
    """
    Main execution function with enhanced error handling and logging
    """
    try:
        # Ensure required directories exist
        Config.create_paths()
        
        # Get or create Spark session with optimized configuration
        spark = SparkSession.builder \
            .appName("FraudDetection-DataPreparation") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "200") \
            .getOrCreate()
        
        # Execute pipeline
        raw_df = load_transaction_data(spark)
        validation_metrics = validate_data(raw_df)
        processed_df, cleaning_metrics = clean_data(raw_df, validation_metrics)
        save_processed_data(processed_df, validation_metrics, cleaning_metrics)
        
        logger.info("Data preparation pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        #spark.stop()
        pass
    
if __name__ == "__main__":
    main()