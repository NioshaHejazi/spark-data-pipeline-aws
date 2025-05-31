from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, regexp_replace, split

# Initialize Spark
spark = SparkSession.builder.appName("TextPreprocessing").getOrCreate()

# Load text file into DataFrame (one line per row)
df = spark.read.text("sample_text.txt").toDF("line")

# Clean text: lowercase, remove punctuation
cleaned = df.withColumn("clean_line", lower(regexp_replace("line", r"[^a-zA-Z\s]", "")))

# Tokenize: split into words
tokenized = cleaned.withColumn("tokens", split("clean_line", r"\s+"))

# Show the result
tokenized.select("tokens").show(truncate=False)

# Stop Spark
spark.stop()
