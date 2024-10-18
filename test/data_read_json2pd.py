import json
import pandas as pd

with open('test/file.json', "r") as f:
    content = json.load(f)
    print(len(content))

df = pd.DataFrame(content[0])

for i in range(1, len(content)):
    df_ = pd.DataFrame(content[i])
    df = pd.concat([df, df_], ignore_index=True)
print(df)
df.to_pickle('test/scene/plat.pkl')