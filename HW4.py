from pyspark import SparkContext 
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import OneHotEncoder, StringIndexer, StringIndexerModel, StandardScaler, Imputer, VectorAssembler, SQLTransformer
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.linalg import Vectors
spark = SparkSession.builder.master("local").getOrCreate()
sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

# List numerical features & categorical features
target_col = "Cancelled"
highest_cols = ["Month", "DayOfWeek", "UniqueCarrier", "FlightNum", "Origin", "Dest"]
mean_cols = ["Year", "DayofMonth", "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "Distance", "TaxiIn", "TaxiOut"]

def load(path):
  # Load DataFrame
  df = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load(path)

  # Select useful columns (drop columns that should not be known 
  # before the flight take place) 
  df = df.select([target_col] + mean_cols + highest_cols)

  # Impute numerical features
  for col in mean_cols:
      df = df.withColumn(col, df[col].cast('double'))
      mu = df.select(col).agg({col:'mean'}).collect()[0][0]
      df = df.withColumn(col, F.when(df[col].isNull(), mu).otherwise(df[col]))
  df = df.withColumn('label', df[target_col].cast('double'))
  df = df.filter(df['label'].isNotNull())

  # Impute categorical features
  for col in highest_cols:
      frq = df.select(col).groupby(col).count().orderBy('count', ascending=False).limit(1).collect()[0][0]
      df = df.withColumn(col, F.when((df[col].isNull() | (df[col] == '')), frq).otherwise(df[col]))

  # Assure there is no missing values
  for col in mean_cols + highest_cols + ['label']:
      assert df.filter(df[col].isNull()).count() == 0, "Column '{}' exists NULL value(s)".format(col)
      assert df.filter(df[col] == '').count() == 0, "Column '{}' exists empty string(s)".format(col)
  return df


def preprocess_train(df):
  # String Indexing for categorical features
  indexers = [StringIndexer(inputCol=col, outputCol="{}_idx".format(col)) for col in highest_cols]   
  # One-hot encoding for categorical features
  encoders = [OneHotEncoder(inputCol="{}_idx".format(col), outputCol="{}_oh".format(col)) for col in highest_cols]
  # Concat Feature Columns
  assembler = VectorAssembler(inputCols = mean_cols + ["{}_oh".format(col) for col in highest_cols], outputCol = "_features")
  
  # Standardize Features
  scaler = StandardScaler(inputCol='_features', outputCol='features', withStd=True, withMean=False)
  preprocess = Pipeline(stages = indexers + encoders + [assembler, scaler]).fit(df)
  return preprocess

def preprocess_test(df, preprocess):
  dic = {x._java_obj.getInputCol():
    [lab for lab in x._java_obj.labels()] for x in preprocess.stages if isinstance(x, StringIndexerModel)}

  # Filter out unseen labels 
  for col in highest_cols:
    df = df.filter(F.col(col).isin(dic[col]))
  
  # Assure there is no unseen values
  for col in highest_cols:
    assert df.filter(F.col(col).isin(dic[col]) == False).count() == 0, "Column '{}' exists unseen label(s)".format(col)
  df = preprocess.transform(df)
  return df


def evaluate(predictionAndLabels):
    log = {}
    # Show Validation Score (AUROC)
    evaluator = BinaryClassificationEvaluator(metricName='areaUnderROC')
    log['AUROC'] = "%f" % evaluator.evaluate(predictionAndLabels)    
    print("Area under ROC = {}".format(log['AUROC']))

    # Show Validation Score (AUPR)
    evaluator = BinaryClassificationEvaluator(metricName='areaUnderPR')
    log['AUPR'] = "%f" % evaluator.evaluate(predictionAndLabels)
    print("Area under PR = {}".format(log['AUPR']))

    # Metrics
    predictionRDD = predictionAndLabels.select(['label', 'prediction']).rdd.map(lambda line: (line[1], line[0]))
    metrics = MulticlassMetrics(predictionRDD)

    # Confusion Matrix
    print("Confusion Matrix")
    print(metrics.confusionMatrix().toArray())

    # Overall statistics
    log['precision'] = "%s" % metrics.precision()
    log['recall'] = "%s" % metrics.recall()
    log['F1 Measure'] = "%s" % metrics.fMeasure()
    print("[Overall]\tprecision = %s | recall = %s | F1 Measure = %s" % (log['precision'], log['recall'], log['F1 Measure']))

    # Statistics by class
    labels = [0.0, 1.0]
    for label in sorted(labels):
        log[label] = {}
        log[label]['precision'] = "%s" % metrics.precision(label)
        log[label]['recall'] = "%s" % metrics.recall(label)
        log[label]['F1 Measure'] = "%s" % metrics.fMeasure(label, beta=1.0)
        print("[Class %s]\tprecision = %s | recall = %s | F1 Measure = %s" % (label, log[label]['precision'], log[label]['recall'], log[label]['F1 Measure']))
    return log
    

print("Load Data")
df = load('/HW4/input/train')
#df_test = load("/HW4/input/test")  
"""
print("train preprocess")
preprocess = preprocess_train(df)   
df = preprocess.transform(df) 
preprocess.write().overwrite().save("/HW4/output/preprocess")
"""
""" read preprocess """
print("read preprocess")
preprocess = PipelineModel.load("/HW4/output/preprocess")
df = preprocess.transform(df)

# testing Pre-Process
#print("test preprocess")
#preprocess = PipelineModel.load("/HW4/output/preprocess")
#df_test = preprocess_test(df_test, preprocess)

# GBTClassifier
print("start training")
#gbt = GBTClassifier(maxIter=10)#10^5
#paramGrid = (ParamGridBuilder().addGrid(gbt.maxDepth, [2, 4, 6]).addGrid(gbt.maxBins, [20, 60]).addGrid(gbt.maxIter, [10, 20]).build())
#cvModel = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=BinaryClassificationEvaluator(metricName='areaUnderPR'), numFolds=10).fit(df)

lr = LogisticRegression(maxIter=5)
cvModel = CrossValidator(estimator=Pipeline(stages = [lr]),
            estimatorParamMaps=ParamGridBuilder() \
                                .addGrid(lr.regParam, [0.1, 0.01]) \
                                .build(),
            evaluator=BinaryClassificationEvaluator(metricName='areaUnderPR'),
            numFolds=10)
split = ['2000', '2001', '2002', '2003', '2004']
for i in split:
    print(i)
    df_train = df.filter(F.col('Year')==i)
    cvModel.fit(df_train)
    cvModel.bestModel.write().overwrite().save("/HW4/output/model")

print("Evaluate training")
predictionAndLabels = cvModel.transform(df)
log = evaluate(predictionAndLabels)
with open('./log/train.json'.format(), 'w') as f:
    json.dump(log, f)

print("start testing")
df_test = preprocess_test(df_test, preprocess)
cvModel = PipelineModel.load("./HW4/output/model")
predictionAndLabels = cvModel.transform(df_test)
predictionAndLabels.write.format('json').write().overwrite().save("/HW4/output/result")

print("Evaluate testing")
log = evaluate(predictionAndLabels)
with open('./log/test.json', 'w') as f:
    json.dump(log, f)
