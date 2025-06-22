from pyspark.sql.functions import expr, udf, col, when, size, lit, aggregate, array_max
from pyspark.sql import functions as F

from pyspark.sql.types import StructType, StructField, ArrayType, StringType, FloatType

from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device = device)

###################
# Soft Skills
###################
@udf(StructType([
    StructField("matches", ArrayType(StringType()), nullable=True),
    StructField("scores", ArrayType(FloatType()), nullable=True)
]))
def match_soft_skills_array(arr1, arr2):
    if arr1 and arr2:
        arr1_vectors = embedding_model.encode(arr1, normalize_embeddings=True)
        arr2_vectors = embedding_model.encode(arr2, normalize_embeddings=True)

        similarities = cosine_similarity(arr2_vectors, arr1_vectors)

        matches = []
        scores = []

        for i, row in enumerate(similarities):
            best_idx = np.argmax(row)
            best_score = row[best_idx]

            if best_score > 0:
                matches.append(arr1[best_idx])
                scores.append(float(best_score))
            else:
                matches.append(None)
                scores.append(None)

        return matches, scores
    return None, None

def create_soft_skills_column(df):
    # Create intermediate soft skills columns
    df = df.withColumn("soft_skills_compare", match_soft_skills_array("soft_skills", "jd_soft_skills"))
    
    # Get mean soft skills similarity score
    df = df.withColumn("soft_skills_mean_score", when(
        (col("soft_skills_compare.scores").isNotNull()) & (size(col("soft_skills_compare.scores")) > 0),
        aggregate(
            col("soft_skills_compare.scores"),
            lit(0.0),
            lambda acc, x: acc + x
        ) / size(col("soft_skills_compare.scores"))
    ).otherwise(None))

    # Get max soft skills score
    df = df.withColumn(
        "soft_skills_max_score",
        when(
            (col("soft_skills_compare.scores").isNotNull()) & (size(col("soft_skills_compare.scores")) > 0),
            array_max(col("soft_skills_compare.scores"))
        ).otherwise(None)
    )

    # Get skills match count
    df = df.withColumn(
        "soft_skills_count",
        when(
            (col("soft_skills_compare.scores").isNotNull()),
            size(expr("filter(soft_skills_compare.scores, x -> x > 0.8)"))
        ).otherwise(0)
    )

    # Get skills match ratio
    df = df.withColumn("jd_soft_skills_count", size(df["jd_soft_skills"]))
    df = df.withColumn(
        f"soft_skills_ratio",
        when(col(f"jd_soft_skills_count") != 0, col(f"soft_skills_count") / col(f"jd_soft_skills_count")).otherwise(None)
    )

    # Drop intermediate columns
    df = df.drop("soft_skills", "jd_soft_skills", "jd_soft_skills_count", "soft_skills_compare")

    return df

###################
# Hard Skills
###################
def _create_hard_skills_column(df, granularity):
    # Get count of skills in JD that are mentioned in resume
    df = df.withColumn(
        f"hard_skills_{granularity}_count",
        expr(f"aggregate(filter(jd_hard_skills_{granularity}, x -> array_contains(hard_skills_{granularity}, x)), 0, (acc, x) -> acc + 1)")
    )
    
    # Get count of skills in JD
    df = df.withColumn(f"jd_hard_skills_{granularity}_count", size(df[f"jd_hard_skills_{granularity}"]))

    # Create ratio column
    df = df.withColumn(
        f"hard_skills_{granularity}_ratio",
        when(col(f"jd_hard_skills_{granularity}_count") != 0, col(f"hard_skills_{granularity}_count") / col(f"jd_hard_skills_{granularity}_count")).otherwise(None)
    )

    # Drop intermediate columns
    df = df.drop(f"jd_hard_skills_{granularity}_count", f"jd_hard_skills_{granularity}", f"hard_skills_{granularity}")

    return df

def create_hard_skills_general_column(df):
    return _create_hard_skills_column(df, "general")

def create_hard_skills_specific_column(df):
    return _create_hard_skills_column(df, "specific")



