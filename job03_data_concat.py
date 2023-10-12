import pandas as pd
import glob
import datetime

data_path = sorted(glob.glob("./crawling_data/*"))
print(data_path)

df = pd.DataFrame()
for path in data_path:
    df_temp = pd.read_csv(path)
    df = pd.concat([df, df_temp])
print(df.head())

print(df["category"].value_counts())
df.info()
df.to_csv(
    "./crawling_data/naver_news_titles_{}.csv".format(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ),
    index=False,
)
