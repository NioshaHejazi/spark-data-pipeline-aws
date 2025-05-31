from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, regexp_replace, split, concat_ws, col
import joblib

# Load model + vectorizer
clf = joblib.load("models/text_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Start Spark
spark = SparkSession.builder.appName("ClassifyText").getOrCreate()

# Load new input text
df = spark.read.text("jobs/sample_text.txt").toDF("line")

# Clean + tokenize (same as before)
df_clean = df.withColumn("clean_line", lower(regexp_replace("line", r"[^a-zA-Z\s]", "")))
df_tokens = df_clean.withColumn("tokens", split("clean_line", r"\s+"))

# Combine tokens into one string per row
df_ready = df_tokens.withColumn("joined", concat_ws(" ", col("tokens")))
df_ready = df_ready.filter(col("joined") != "")


# Collect joined text for inference
text_list = df_ready.select("joined").rdd.flatMap(lambda x: x).collect()

# Vectorize and predict
X_vec = vectorizer.transform(text_list)
preds = clf.predict(X_vec)

label_map = {0: "comp.graphics", 1: "sci.space"}

# Print predictions
for text, label in zip(text_list, preds):
    print(f"\n📄 Text: {text}\n🔍 Predicted Label: {label_map[label]}")


spark.stop()
