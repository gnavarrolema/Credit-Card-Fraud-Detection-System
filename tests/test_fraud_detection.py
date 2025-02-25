import unittest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
from pyspark.sql.functions import col, lit
import datetime


# Tests para DataPreparation

class TestDataPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .appName("FraudDetection-Tests") \
            .getOrCreate()
        
        # Create a robust test dataset with variations
        data = [
            # Valid Data
            ("tx1", "RETAIL", 100.50, datetime.datetime(2023, 1, 1, 10, 0, 0), 1),
            ("tx2", "ONLINE", -50.00, datetime.datetime(2023, 1, 1, 10, 5, 0), 0),
            ("tx3", "TRAVEL", 200.75, datetime.datetime(2023, 1, 1, 10, 10, 0), 1),
            # Data with null values
            ("tx4", None, 150.00, datetime.datetime(2023, 1, 1, 10, 15, 0), 0),
            # Data with invalid timestamps
            ("tx5", "RETAIL", 300.00, None, 1),
            # More variations
            ("tx6", "ONLINE", 0.00, datetime.datetime(2023, 1, 1, 10, 20, 0), 0),
        ]
        
        cls.schema = StructType([
            StructField("transaction_id", StringType(), False),
            StructField("merchant_category", StringType(), True),
            StructField("amount", DoubleType(), False),
            StructField("timestamp", TimestampType(), True),
            StructField("is_fraud", IntegerType(), False)
        ])
        
        cls.test_df = cls.spark.createDataFrame(data, cls.schema)
    
    def test_validate_data(self):
        from data_preparation import validate_data
        validation_metrics = validate_data(self.test_df)
        
        # Verify null metrics
        self.assertEqual(validation_metrics["null_metrics"]["merchant_category"]["null_count"], 1)
        self.assertEqual(validation_metrics["null_metrics"]["timestamp"]["null_count"], 1)
        
        # Verify metrics of invalid values
        self.assertEqual(validation_metrics["invalid_amounts"], 1)  
    
    def test_clean_data(self):
        from data_preparation import clean_data
        cleaned_df, cleaning_metrics = clean_data(self.test_df, {})
        
        # Verify that invalid records have been removed
        self.assertEqual(cleaned_df.count(), 4)  
        
        # Verify cleaning metrics
        self.assertEqual(cleaning_metrics["removed_records"], 2)
        self.assertEqual(cleaning_metrics["nulls_removed"], 2)


# Tests for FeatureEngineering

class TestFeatureEngineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .appName("FraudDetection-Tests") \
            .getOrCreate()
        
        # Create a test dataset with timestamps and multiple categories
        data = [
            ("tx1", "RETAIL", 100.50, datetime.datetime(2023, 1, 1, 10, 0, 0), 1),
            ("tx2", "RETAIL", 150.00, datetime.datetime(2023, 1, 1, 10, 1, 0), 0),
            ("tx3", "ONLINE", 200.75, datetime.datetime(2023, 1, 1, 10, 2, 0), 1),
            ("tx4", "ONLINE", 300.00, datetime.datetime(2023, 1, 1, 10, 3, 0), 0),
            ("tx5", "TRAVEL", 500.00, datetime.datetime(2023, 1, 1, 10, 4, 0), 1),
        ]
        
        schema = StructType([
            StructField("transaction_id", StringType(), False),
            StructField("merchant_category", StringType(), True),
            StructField("amount", DoubleType(), False),
            StructField("timestamp", TimestampType(), True),
            StructField("is_fraud", IntegerType(), False)
        ])
        
        cls.test_df = cls.spark.createDataFrame(data, schema)
    
    def test_calculate_velocity_features(self):
        from feature_engineering import FeatureConfig, FeatureEngineer
        config = FeatureConfig()
        engineer = FeatureEngineer(self.spark, config)
        velocity_df = engineer.calculate_velocity_features(self.test_df)
        
        # Verify specific values for speed features
        row = velocity_df.filter("transaction_id = 'tx1'").select("tx_count_short").collect()[0]
        self.assertEqual(row["tx_count_short"], 2)  
    
    def test_create_merchant_profiles(self):
        from feature_engineering import FeatureConfig, FeatureEngineer
        config = FeatureConfig()
        engineer = FeatureEngineer(self.spark, config)
        profile_df = engineer.create_merchant_profiles(self.test_df)
        
        # Verify specific values for merchant profiles
        retail_avg = profile_df.filter("merchant_category = 'RETAIL'").select("merchant_avg_amount").collect()[0][0]
        self.assertAlmostEqual(retail_avg, 125.25, places=2)  


# Tests para ModelTraining

class TestModelTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .appName("FraudDetection-Tests") \
            .getOrCreate()
        
        # Create a larger test dataset for training
        data = [
            ("tx1", 100.50, 1.0, 2.0, 1),
            ("tx2", 150.00, 0.5, 1.0, 0),
            ("tx3", 200.75, 1.5, 3.0, 1),
            ("tx4", 300.00, 0.8, 1.5, 0),
            ("tx5", 500.00, 2.0, 4.0, 1),
        ]
        
        cls.schema = StructType([
            StructField("transaction_id", StringType(), False),
            StructField("amount", DoubleType(), False),
            StructField("velocity_feature1", DoubleType(), False),
            StructField("velocity_feature2", DoubleType(), False),
            StructField("is_fraud", IntegerType(), False)
        ])
        
        cls.test_df = cls.spark.createDataFrame(data, cls.schema)
    
    def test_handle_class_imbalance(self):
        from model_training import ModelConfig, ModelTrainer
        config = ModelConfig()
        trainer = ModelTrainer(self.spark, config)
        weighted_df = trainer.handle_class_imbalance(self.test_df)
        
        # Verify the calculated weights
        total = 5
        fraud_count = 3
        expected_fraud_weight = total / (2 * fraud_count)
        fraud_weight = weighted_df.filter("is_fraud = 1").select("class_weight").first()[0]
        self.assertAlmostEqual(fraud_weight, expected_fraud_weight, places=2)
    
    def test_prepare_training_data(self):
        from model_training import ModelConfig, ModelTrainer
        config = ModelConfig()
        trainer = ModelTrainer(self.spark, config)
        train_df, test_df = trainer.prepare_training_data(self.test_df)
        
        # Verify the division
        self.assertEqual(train_df.count(), 4)  
        self.assertEqual(test_df.count(), 1)   
    
    def test_train_and_evaluate_model(self):
        from model_training import ModelConfig, ModelTrainer
        config = ModelConfig()
        trainer = ModelTrainer(self.spark, config)
        model, metrics = trainer.train_and_evaluate_model(self.test_df)
        
        # Verify the model metrics
        self.assertGreater(metrics["accuracy"], 0.5)
        self.assertGreater(metrics["f1_score"], 0.5)

def run_tests():
    """Ejecuta todos los tests unitarios"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreparation))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureEngineering))
    suite.addTests(loader.loadTestsFromTestCase(TestModelTraining))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == "__main__":
    run_tests()