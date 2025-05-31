from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

# Initialize Spark
spark = SparkSession.builder.appName("TFIDFEmbedding").getOrCreate()

# Load cleaned and tokenized data (run preprocess.py first!)
df = spark.read.text("jobs/sample_text.txt").toDF("line")

# Clean and tokenize inline
from pyspark.sql.functions import lower, regexp_replace, split
df_clean = df.withColumn("clean_line", lower(regexp_replace("line", r"[^a-zA-Z\s]", "")))
df_tokens = df_clean.withColumn("tokens", split("clean_line", r"\s+"))

# TF step: term frequencies
hashing_tf = HashingTF(inputCol="tokens", outputCol="rawFeatures", numFeatures=100)
tf_df = hashing_tf.transform(df_tokens)

# IDF step: scale by inverse doc frequency
idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(tf_df)
tfidf_df = idf_model.transform(tf_df)

# Show the result
tfidf_df.select("tokens", "features").show(truncate=False)

spark.stop()
