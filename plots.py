import os
import duckdb
import seaborn as sns
import matplotlib.pyplot as plt

cm = 1 / 2.54

remaps = {
    "evaluation": {0:"silhouette", 1:"F-statistic", 2:"binary info", 3:"information gain", 'multiclass info':"information gain"},
    "extraction": {6: "random samples", 7: "FSS", 8:"clustering"},
    "model": {0: "Logistic Regression", 3: "Gaussian Process", 4: "Decision Tree", 5: "Random Forest"}
}

# Figure configs
evaluation_colors = {"silhouette": "green", "fstat": "orange", "information gain": "blue"}
evaluation_order = ["information gain", "silhouette", "fstat"]
extraction_colors = {"clustering": "green", "FSS": "orange", "random samples": "blue"}
extraction_order = ["random samples", "clustering", "FSS"]

db = duckdb.connect()
db.sql("CREATE VIEW candidates AS select * FROM 'db/candidates.csv'");
db.sql("CREATE VIEW dataset AS select * FROM 'db/dataset.csv'");
db.sql("CREATE VIEW classification AS select * FROM 'db/classification.csv'");
db.sql("CREATE VIEW num_candidates AS SELECT dataset, method as extraction, count(*) as n_candidates from candidates group by all")

db.sql("""
    CREATE VIEW valid_datasets AS
    WITH
        counts AS (
            SELECT dataset, count(distinct extraction) as n_extraction 
            FROM classification 
            WHERE evaluation = 'silhouette' 
            GROUP BY ALL
        )
    SELECT dataset
    FROM counts
    WHERE n_extraction = 3 -- keep datasets where all three methods have been tested
    """)

print("="*60)
print("Many times the clustering has a better accuracy, but when the")
print("random selection works better, then it does so with fewer candidates")
db.sql("""
    WITH
        base AS (
            SELECT dataset, k, extraction, n_candidates, accuracy 
            FROM classification 
                NATURAL JOIN num_candidates 
                NATURAL JOIN valid_datasets
            WHERE evaluation = 'silhouette' 
              AND extraction != 7 -- exclude FSS
        ),
        pivot_table AS (
            PIVOT base
            ON extraction IN (6, 8)
            USING avg(n_candidates) AS n_candidates, avg(accuracy) AS accuracy
            GROUP BY k, dataset
        ),
        situations AS (
            SELECT k, CASE
                WHEN "6_accuracy" < "8_accuracy" THEN 'clustering_better'
                WHEN "6_n_candidates" < "8_n_candidates" THEN 'random_better_fewer_candidates'
                WHEN "6_n_candidates" >= "8_n_candidates" THEN 'random_better_more_candidates'
            END AS situation
            FROM pivot_table
        )
    SELECT k, situation, count(*) 
    FROM situations
    GROUP BY ALL
    ORDER BY ALL
""").show()

print("These are the results to be plotted")
res_classification = db.sql("""
    SELECT k, extraction, n_candidates, accuracy 
    FROM classification 
        NATURAL JOIN num_candidates 
        NATURAL JOIN valid_datasets
    WHERE evaluation = 'silhouette' 
    """)
res_classification.show()
df = res_classification.to_df()

if not os.path.isdir("imgs"):
    os.mkdir("imgs")

fig, ax = plt.subplots(figsize=(14 * cm, 5 * cm))
sns.boxplot(
    df.replace(remaps),
    x="k",
    y="accuracy",
    hue="extraction",
    showfliers=False,
    palette=extraction_colors,
    hue_order=extraction_order,
    ax=ax
)
ax.legend(loc="lower center", fancybox=True, framealpha=0.9, ncol=3)
ax.set(ylim=(0.36, 1.01))
fig.tight_layout()
plt.savefig("imgs/matteo_extraction_comparison.png")

# fig, ax = plt.subplots(figsize=(14 * cm, 5 * cm))
g = sns.FacetGrid(
    df.replace(remaps),
    col="k",
    col_wrap=3
)
g.map_dataframe(
    sns.scatterplot,
    x="n_candidates",
    y="accuracy",
    hue="extraction",
    palette=extraction_colors,
    hue_order=extraction_order,
)
g.add_legend()
plt.tight_layout()
plt.savefig("imgs/matteo_extraction_comparison_scatter.png")


