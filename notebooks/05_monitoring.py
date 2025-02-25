from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, floor, count, avg, sum, stddev,
    min, max, percentile_approx, when,
    date_trunc, current_timestamp, datediff,
    expr, window, element_at, lit
)
from pyspark.sql.types import DoubleType, IntegerType, FloatType
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import logging
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta
import os

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Paths
SCORED_DATA_PATH = "/FileStore/fraud_detection/results/scored_transactions"
MONITORING_OUTPUT_PATH = "/FileStore/fraud_detection/monitoring"
FEATURE_CONFIG_PATH = "/FileStore/fraud_detection/data/features/feature_metadata.json"

# Monitoring configuration
MONITORING_WINDOW_DAYS = 7
ALERT_THRESHOLDS = {
    "auc_drop": 0.05,
    "prediction_drift": 0.1,
    "data_quality": 0.01
}

def load_monitoring_data(spark: SparkSession) -> Dict[str, DataFrame]:
    """
    Load scored transactions and reference data for monitoring.
    Use a fixed timestamp for consistency in a distributed environment.
    """
    try:
        logger.info(f"Loading transactions from {SCORED_DATA_PATH}")
        current_time = datetime.now()
        cutoff_recent = current_time - timedelta(days=MONITORING_WINDOW_DAYS)
        cutoff_reference = current_time - timedelta(days=14)

        # Load recent and reference data
        recent_data = spark.read.format("delta").load(SCORED_DATA_PATH) \
            .where(col("scoring_timestamp") >= cutoff_recent)
        reference_data = spark.read.format("delta").load(SCORED_DATA_PATH) \
            .where((col("scoring_timestamp") < cutoff_recent) & (col("scoring_timestamp") >= cutoff_reference))

        # Verify data availability
        recent_count = recent_data.count()
        reference_count = reference_data.count()
        if recent_count == 0 or reference_count == 0:
            logger.warning("There is not enough data for monitoring. Verify the data load.")

        logger.info(f"Loaded {recent_count} recent records and {reference_count} reference records.")
        return {"recent": recent_data.cache(), "reference": reference_data.cache()}
    
    except Exception as e:
        logger.error(f"Error loading monitoring data: {str(e)}")
        raise

def calculate_performance_metrics(df: DataFrame, reference_df: Optional[DataFrame] = None) -> Dict:
    """
    Calculate model performance metrics, checking for the presence of labels.
    Filter out null values to avoid errors in the evaluators.
    """
    try:
        logger.info("Calculating performance metrics")
        metrics = {}

        # Check for the presence of labels
        if "is_fraud" not in df.columns:
            metrics["note"] = "No ground truth labels available; skipping performance metrics."
            logger.info("No ground truth labels found; skipping performance metrics.")
            return metrics
        
        # Filter out null values in critical columns
        df = df.filter(col("probability").isNotNull() & col("prediction").isNotNull() & col("is_fraud").isNotNull())
        if reference_df is not None:
            reference_df = reference_df.filter(
                col("probability").isNotNull() & col("prediction").isNotNull() & col("is_fraud").isNotNull()
            )

        evaluators = {
            "areaUnderROC": BinaryClassificationEvaluator(
                labelCol="is_fraud", rawPredictionCol="probability", metricName="areaUnderROC"
            ),
            "areaUnderPR": BinaryClassificationEvaluator(
                labelCol="is_fraud", rawPredictionCol="probability", metricName="areaUnderPR"
            )
        }

        # Calculate metrics for recent data
        for metric_name, evaluator in evaluators.items():
            metrics[metric_name] = evaluator.evaluate(df)

        # Compare with reference data if available
        if reference_df is not None:
            for metric_name, evaluator in evaluators.items():
                ref_metric = evaluator.evaluate(reference_df)
                metrics[f"{metric_name}_reference"] = ref_metric
                metrics[f"{metric_name}_change"] = metrics[metric_name] - ref_metric
                if metrics[f"{metric_name}_change"] < -ALERT_THRESHOLDS["auc_drop"]:
                    logger.warning(f"Warning: {metric_name} dropped by {abs(metrics[f'{metric_name}_change']):.3f}")

        # Confusion matrix
        confusion_matrix = df.groupBy("is_fraud", "prediction").count().orderBy("is_fraud", "prediction")
        metrics["confusion_matrix"] = confusion_matrix.collect()
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {str(e)}")
        raise

def analyze_feature_distributions(
    df: DataFrame, reference_df: Optional[DataFrame] = None, feature_list: Optional[List[str]] = None
) -> Dict:
    """
    Analyze the distribution of specific features with a single aggregation.
    If feature_list is not provided, load it from feature_config.json.
    """
    try:
        logger.info("Analyzing feature distributions")
        
        # Load features from feature_config if feature_list is not provided
        if feature_list is None:
            with open(FEATURE_CONFIG_PATH.replace("/FileStore", "/dbfs"), "r") as f:
                feature_config = json.load(f)
            feature_list = [col for group in feature_config["feature_groups"].values() for col in group]

        # Filter features present in the DataFrame
        feature_list = [f for f in feature_list if f in df.columns]
        if not feature_list:
            logger.warning("No features found in DataFrame for distribution analysis.")
            return {}

        # Calculate statistics in a single aggregation
        stats_df = df.select([
            avg(col(f)).alias(f"{f}_mean"),
            stddev(col(f)).alias(f"{f}_stddev"),
            min(col(f)).alias(f"{f}_min"),
            max(col(f)).alias(f"{f}_max"),
            percentile_approx(f, [0.25, 0.5, 0.75]).alias(f"{f}_percentiles")
        for f in feature_list]).collect()[0]
        
        distribution_metrics = {}
        for feature in feature_list:
            distribution_metrics[feature] = {
                "current": {
                    "mean": float(stats_df[f"{feature}_mean"]) if stats_df[f"{feature}_mean"] is not None else None,
                    "stddev": float(stats_df[f"{feature}_stddev"]) if stats_df[f"{feature}_stddev"] is not None else None,
                    "min": float(stats_df[f"{feature}_min"]) if stats_df[f"{feature}_min"] is not None else None,
                    "max": float(stats_df[f"{feature}_max"]) if stats_df[f"{feature}_max"] is not None else None,
                    "percentiles": [float(p) for p in stats_df[f"{feature}_percentiles"]] if stats_df[f"{feature}_percentiles"] is not None else None
                }
            }
        
        # Analyze drifts with reference data
        if reference_df is not None:
            ref_stats_df = reference_df.select([
                avg(col(f)).alias(f"{f}_mean"),
                stddev(col(f)).alias(f"{f}_stddev")
            for f in feature_list if f in reference_df.columns]).collect()[0]
            
            for feature in feature_list:
                if feature in reference_df.columns:
                    ref_mean = float(ref_stats_df[f"{feature}_mean"]) if ref_stats_df[f"{feature}_mean"] is not None else None
                    ref_stddev = float(ref_stats_df[f"{feature}_stddev"]) if ref_stats_df[f"{feature}_stddev"] is not None else None
                    distribution_metrics[feature]["reference"] = {
                        "mean": ref_mean,
                        "stddev": ref_stddev
                    }
                    # Compute drift, avoiding division by zero
                    if ref_stddev and ref_stddev != 0 and distribution_metrics[feature]["current"]["mean"] is not None:
                        mean_diff = abs(distribution_metrics[feature]["current"]["mean"] - ref_mean) / ref_stddev
                    else:
                        mean_diff = 0 if distribution_metrics[feature]["current"]["mean"] == ref_mean else float('inf')
                    distribution_metrics[feature]["drift_score"] = mean_diff
                    
                    if mean_diff > ALERT_THRESHOLDS["prediction_drift"]:
                        logger.warning(f"Warning: Feature '{feature}' shows significant drift (score: {mean_diff:.3f})")
        
        return distribution_metrics
    
    except Exception as e:
        logger.error(f"Error analyzing feature distributions: {str(e)}")
        raise

def analyze_prediction_distribution(df: DataFrame, reference_df: Optional[DataFrame] = None) -> Dict:
    """
    Analyze the distribution of predictions using native functions instead of UDFs.
    """
    try:
        logger.info("Analyzing prediction distribution")
        
        # Extract fraud probability without a UDF
        df = df.withColumn("fraud_probability", element_at(col("probability"), 2)) \
               .withColumn("prob_bin", floor(col("fraud_probability") * 10) / 10)
        
        current_dist = df.groupBy("prob_bin") \
            .agg(count("*").alias("count")) \
            .orderBy("prob_bin")
        
        distribution_metrics = {"current": current_dist.collect()}
        
        if reference_df is not None:
            reference_df = reference_df.withColumn("fraud_probability", element_at(col("probability"), 2)) \
                                       .withColumn("prob_bin", floor(col("fraud_probability") * 10) / 10)
            ref_dist = reference_df.groupBy("prob_bin") \
                .agg(count("*").alias("count")) \
                .orderBy("prob_bin")
            distribution_metrics["reference"] = ref_dist.collect()
        
        return distribution_metrics
    
    except Exception as e:
        logger.error(f"Error analyzing prediction distribution: {str(e)}")
        raise

def monitor_data_quality(df: DataFrame) -> Dict:
    """
    Monitor data quality by counting nulls and detecting duplicates.
    """
    try:
        logger.info("Monitoring data quality")
        
        quality_metrics = {}
        total_count = df.count()
        
        for column in df.columns:
            null_count = df.filter(col(column).isNull()).count()
            null_percentage = null_count / total_count if total_count > 0 else 0
            quality_metrics[f"{column}_nulls"] = {
                "count": null_count,
                "percentage": null_percentage
            }
            if null_percentage > ALERT_THRESHOLDS["data_quality"]:
                logger.warning(f"Warning: Column '{column}' has {null_percentage:.2%} null values.")
        
        # Detect duplicates using a composite key
        duplicates = df.groupBy("transaction_id", "scoring_timestamp").count().filter(col("count") > 1)
        duplicate_count = duplicates.count()
        quality_metrics["duplicates"] = {
            "count": duplicate_count,
            "percentage": duplicate_count / total_count if total_count > 0 else 0
        }
        
        return quality_metrics
    
    except Exception as e:
        logger.error(f"Error monitoring data quality: {str(e)}")
        raise

def generate_visualizations(df: DataFrame, monitoring_results: Dict) -> None:
    """
    Generate visualizations using Databricks capabilities if running in that environment.
    """
    try:
        if os.environ.get("DATABRICKS_RUNTIME_VERSION"):  # Detect if running on Databricks
            from pyspark.sql.functions import col
            logger.info("Generating visualizations for the Databricks notebook")
            # Histogram of probabilities
            display(df.groupBy("prob_bin").count().orderBy("prob_bin"))
            # You can add more visualizations as needed
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")

def save_monitoring_results(
    performance_metrics: Dict,
    distribution_metrics: Dict,
    prediction_metrics: Dict,
    quality_metrics: Dict
) -> None:
    """
    Save the results with a unique timestamp and partition the Delta table by date.
    """
    try:
        logger.info(f"Saving monitoring results to {MONITORING_OUTPUT_PATH}")
        
        monitoring_results = {
            "timestamp": datetime.now().isoformat(),
            "performance": performance_metrics,
            "distributions": distribution_metrics,
            "predictions": prediction_metrics,
            "data_quality": quality_metrics
        }
        
        # Save JSON with a unique timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path_json = f"{MONITORING_OUTPUT_PATH}/results_{timestamp_str}.json".replace("/FileStore", "/dbfs")
        with open(output_path_json, "w") as f:
            json.dump(monitoring_results, f, indent=2)
        
        # Save to a historical Delta table with partitioning
        spark = SparkSession.builder.getOrCreate()
        monitoring_df = spark.createDataFrame([
            (
                current_timestamp(),
                json.dumps(performance_metrics),
                json.dumps(distribution_metrics),
                json.dumps(prediction_metrics),
                json.dumps(quality_metrics)
            )
        ], ["timestamp", "performance", "distributions", "predictions", "data_quality"])
        
        monitoring_df.write \
            .format("delta") \
            .mode("append") \
            .partitionBy("timestamp") \
            .save(f"{MONITORING_OUTPUT_PATH}/history")
        
        logger.info("Monitoring results saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving monitoring results: {str(e)}")
        raise

def main():
    """Main execution function."""
    try:
        spark = SparkSession.builder \
            .appName("FraudDetection-Monitoring") \
            .getOrCreate()
        
        data_dict = load_monitoring_data(spark)
        
        performance_metrics = calculate_performance_metrics(
            data_dict["recent"],
            data_dict["recent"] if "is_fraud" in data_dict["recent"].columns else data_dict["reference"]
        )
        
        distribution_metrics = analyze_feature_distributions(
            data_dict["recent"],
            data_dict["reference"]
        )
        
        prediction_metrics = analyze_prediction_distribution(
            data_dict["recent"],
            data_dict["reference"]
        )
        
        quality_metrics = monitor_data_quality(data_dict["recent"])
        
        save_monitoring_results(
            performance_metrics,
            distribution_metrics,
            prediction_metrics,
            quality_metrics
        )
        
        # Generate visualizations if running on Databricks
        if "fraud_probability" in data_dict["recent"].columns:
            generate_visualizations(data_dict["recent"], {
                "performance": performance_metrics,
                "distributions": distribution_metrics,
                "predictions": prediction_metrics,
                "data_quality": quality_metrics
            })
        
        logger.info("Monitoring pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in the monitoring pipeline: {str(e)}")
        raise
    finally:
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    main()