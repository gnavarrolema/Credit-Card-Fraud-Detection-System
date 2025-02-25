import mlflow
import mlflow.spark
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, when, expr, date_trunc
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier,
    GBTClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler
)
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics
import logging
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import builtins  # para usar builtins.sum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model training."""
    INPUT_PATH: str = "/FileStore/fraud_detection/data/features"
    MODEL_OUTPUT_PATH: str = "/FileStore/fraud_detection/models/fraud_detector"
    EXPERIMENT_NAME: str = "/Shared/fraud_detection"
    RUN_NAME_PREFIX: str = "fraud_detection_model"
    
    MODELS: Dict = field(default_factory=lambda: {
        "LogisticRegression": {
            "estimator": LogisticRegression(
                featuresCol="scaled_features",
                labelCol="is_fraud",
                weightCol="class_weight"
            ),
            "paramGrid": {
                "regParam": [0.01, 0.1, 1.0],
                "elasticNetParam": [0.0, 0.5, 1.0],
                "maxIter": [20, 50, 100]
            }
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(
                featuresCol="scaled_features",
                labelCol="is_fraud",
                weightCol="class_weight"
            ),
            "paramGrid": {
                "numTrees": [50, 100, 200],
                "maxDepth": [5, 10, 15],
                "maxBins": [32, 64]
            }
        },
        "GradientBoostedTrees": {
            "estimator": GBTClassifier(
                featuresCol="scaled_features",
                labelCol="is_fraud",
                weightCol="class_weight"
            ),
            "paramGrid": {
                "maxIter": [20, 50, 100],
                "maxDepth": [3, 5, 7],
                "stepSize": [0.1, 0.3]
            }
        }
    })
    
    TRAIN_RATIO: float = 0.8
    NUM_FOLDS: int = 5
    RANDOM_SEED: int = 42
    MAX_FEATURES: int = 50
    FEATURE_IMPORTANCE_THRESHOLD: float = 0.01
    
    @classmethod
    def create_paths(cls) -> None:
        """Ensure all necessary directories exist."""
        Path(cls.MODEL_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

class ModelTrainer:
    """Class to handle model training operations."""
    
    def __init__(self, spark: SparkSession, config: ModelConfig):
        self.spark = spark
        self.config = config
        self._setup_mlflow()
    
    def _setup_mlflow(self) -> None:
        mlflow.set_experiment(self.config.EXPERIMENT_NAME)
    
    def load_feature_data(self) -> Tuple[DataFrame, Dict]:
        """Load and validate feature data, including metadata."""
        try:
            logger.info(f"Loading feature data from: {self.config.INPUT_PATH}")
            df = self.spark.read.format("delta").load(self.config.INPUT_PATH)
            with open(f"{self.config.INPUT_PATH}/feature_metadata.json", "r") as f:
                feature_metadata = json.load(f)
            
            if "is_fraud" not in df.columns:
                raise ValueError("Target column 'is_fraud' not found in dataset")
            
            total_records = df.count()
            feature_count = len(df.columns)
            fraud_ratio = df.filter(col("is_fraud") == 1).count() / total_records if total_records else 0
            
            logger.info(f"Loaded {total_records} records with {feature_count} features")
            logger.info(f"Fraud ratio: {fraud_ratio:.2%}")
            
            return df, feature_metadata
        except Exception as e:
            logger.error(f"Error loading feature data: {str(e)}")
            raise
    
    def prepare_training_data(
        self,
        df: DataFrame,
        feature_metadata: Dict
    ) -> Tuple[DataFrame, DataFrame, List[str]]:
        """
        Prepare data for training using a temporal split.
        Devuelve feature_cols para usar en la importancia de características.
        """
        try:
            logger.info("Preparing training data (temporal split)")
            df = df.withColumn("timestamp", col("timestamp").cast("timestamp"))
            split_date = df.select(expr("percentile_approx(timestamp, 0.8) as split_date")).collect()[0]["split_date"]
            logger.info(f"Temporal split date: {split_date}")
            train_data = df.filter(col("timestamp") < split_date)
            test_data = df.filter(col("timestamp") >= split_date)
            logger.info(f"Train count: {train_data.count()} - Test count: {test_data.count()}")
            
            # Convert 'is_fraud' from boolean to integer
            train_data = train_data.withColumn("is_fraud", col("is_fraud").cast("integer"))
            test_data = test_data.withColumn("is_fraud", col("is_fraud").cast("integer"))
            
            # Column-specific imputation
            train_data = train_data.na.fill({"time_since_last_tx": 9999}).na.fill(0)
            test_data = test_data.na.fill({"time_since_last_tx": 9999}).na.fill(0)
            
            # Retrieve feature columns from metadata
            feature_cols = []
            for group_name, group_cols in feature_metadata["feature_groups"].items():
                feature_cols.extend(group_cols)
            
            assembler = VectorAssembler(
                inputCols=feature_cols,
                outputCol="features",
                handleInvalid="skip"
            )
            scaler = StandardScaler(
                inputCol="features",
                outputCol="scaled_features",
                withStd=True,
                withMean=True
            )
            pipeline = Pipeline(stages=[assembler, scaler])
            
            # Fit the pipeline only on train_data and transform both sets
            pipeline_model = pipeline.fit(train_data)
            transformed_train = pipeline_model.transform(train_data)
            transformed_test = pipeline_model.transform(test_data)
            
            logger.info(f"Transformed train count: {transformed_train.count()} - test count: {transformed_test.count()}")
            return transformed_train, transformed_test, feature_cols  
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    def handle_class_imbalance(self, df: DataFrame) -> DataFrame:
        """Apply inverse class weights."""
        try:
            logger.info("Handling class imbalance")
            class_counts = df.groupBy("is_fraud").count().collect()
            total_count = builtins.sum(row["count"] for row in class_counts)
            class_weights = {
                row["is_fraud"]: total_count / (2.0 * row["count"])
                for row in class_counts
            }
            weighted_df = df.withColumn(
                "class_weight",
                when(col("is_fraud") == 1, class_weights[1]).otherwise(class_weights[0])
            )
            logger.info(f"Applied class weights: {class_weights}")
            return weighted_df
        except Exception as e:
            logger.error(f"Error handling class imbalance: {str(e)}")
            raise
    
    def train_and_evaluate_model(
        self,
        model_name: str,
        model_config: Dict,
        train_data: DataFrame,
        test_data: DataFrame,
        feature_cols: List[str]  
    ) -> Dict:
        """Train and evaluate a single model using CrossValidator."""
        try:
            logger.info(f"Training {model_name}")
            with mlflow.start_run(run_name=f"{self.config.RUN_NAME_PREFIX}_{model_name}"):
                mlflow.log_params({
                    "model_type": model_name,
                    "cv_folds": self.config.NUM_FOLDS,
                    "train_samples": train_data.count(),
                    "test_samples": test_data.count()
                })
                paramGrid = ParamGridBuilder()
                for param, values in model_config["paramGrid"].items():
                    paramGrid = paramGrid.addGrid(getattr(model_config["estimator"], param), values)
                evaluator = BinaryClassificationEvaluator(
                    labelCol="is_fraud",
                    metricName="areaUnderROC"
                )
                cv = CrossValidator(
                    estimator=model_config["estimator"],
                    estimatorParamMaps=paramGrid.build(),
                    evaluator=evaluator,
                    numFolds=self.config.NUM_FOLDS,
                    seed=self.config.RANDOM_SEED
                )
                cv_model = cv.fit(train_data)
                best_model = cv_model.bestModel
                predictions = best_model.transform(test_data)
                
                # Metrics
                metrics = {
                    "areaUnderROC": evaluator.evaluate(predictions),
                    "areaUnderPR": BinaryClassificationEvaluator(
                        labelCol="is_fraud",
                        metricName="areaUnderPR"
                    ).evaluate(predictions),
                    "accuracy": MulticlassClassificationEvaluator(
                        labelCol="is_fraud",
                        metricName="accuracy"
                    ).evaluate(predictions),
                    "f1": MulticlassClassificationEvaluator(
                        labelCol="is_fraud",
                        metricName="f1"
                    ).evaluate(predictions)
                }
                
                # Additional metrics for imbalanced classes
                prediction_and_labels = predictions.select("prediction", "is_fraud").rdd.map(lambda r: (float(r[0]), float(r[1])))
                metrics_obj = MulticlassMetrics(prediction_and_labels)
                metrics["recall_class_1"] = metrics_obj.recall(label=1)
                metrics["precision_class_1"] = metrics_obj.precision(label=1)
                
                mlflow.log_metrics(metrics)
                
                # Log feature importance for RF and GBT using feature_cols
                if model_name in ["RandomForest", "GradientBoostedTrees"]:
                    feature_importance = best_model.featureImportances
                    importance_dict = {feature_cols[i]: float(feature_importance[i]) for i in range(len(feature_cols))}
                    mlflow.log_dict(importance_dict, "feature_importance.json")
                
                model_path = f"{self.config.MODEL_OUTPUT_PATH}/{model_name}"
                best_model.write().overwrite().save(model_path)
                mlflow.log_artifacts(model_path, artifact_path="model")
                
                logger.info(f"{model_name} metrics: {metrics}")
                return {
                    "model": best_model,
                    "metrics": metrics,
                    "path": model_path
                }
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            raise
    
    def save_model_results(self, results: Dict) -> None:
        """Save model training results and select the best model."""
        try:
            logger.info("Saving model results")
            best_model = max(
                results.items(),
                key=lambda x: x[1]["metrics"]["areaUnderROC"]
            )
            summary = {
                "timestamp": datetime.now().isoformat(),
                "best_model": {
                    "name": best_model[0],
                    "metrics": best_model[1]["metrics"],
                    "path": best_model[1]["path"]
                },
                "all_models": {
                    name: {
                        "metrics": result["metrics"],
                        "path": result["path"]
                    }
                    for name, result in results.items()
                }
            }
            with open(f"{self.config.MODEL_OUTPUT_PATH}/training_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Model results saved successfully. Best model: {best_model[0]}")
        except Exception as e:
            logger.error(f"Error saving model results: {str(e)}")
            raise


def main():
    """Main execution function."""
    try:
        config = ModelConfig()
        config.create_paths()
        spark = SparkSession.builder \
            .appName("FraudDetection-ModelTraining") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        trainer = ModelTrainer(spark, config)
        df, feature_metadata = trainer.load_feature_data()
        train_data, test_data, feature_cols = trainer.prepare_training_data(df, feature_metadata)  # Obtener feature_cols
        weighted_train_data = trainer.handle_class_imbalance(train_data)
        results = {}
        for model_name, model_cfg in config.MODELS.items():
            model_result = trainer.train_and_evaluate_model(
                model_name,
                model_cfg,
                weighted_train_data,
                test_data,
                feature_cols  
            )
            results[model_name] = model_result
        trainer.save_model_results(results)
        logger.info("Model training pipeline completed successfully")
    except Exception as e:
        logger.error(f"Model training pipeline failed: {str(e)}")
        raise
    finally:
        # spark.stop() if desired
        pass

if __name__ == "__main__":
    main()