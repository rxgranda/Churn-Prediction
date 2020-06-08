from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="review_score", outputCol="indexedLabel").fit(s_df2)
# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
assembler = VectorAssembler(inputCols=['tfidf','sentimentIndex'],outputCol="indexedFeatures")


# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = s_df2.randomSplit([0.7, 0.3])

# Train a GBT model.
rf = RandomForestClassifier(labelCol="review_score", featuresCol="indexedFeatures", numTrees=10)

# Chain indexers and GBT in a Pipeline
pipeline = Pipeline(stages=[tfidf_pipeline, assembler, rf])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "review_score", "indexedFeatures").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="review_score", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

gbtModel = model.stages[2]
print(gbtModel)  # summary only