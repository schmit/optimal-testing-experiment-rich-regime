import pandas as pd
import json, os

from experiments import summarize

import click


def open_files(directory):
    data = []

    for root, dirs, files in os.walk(directory):
        print("Found {} files".format(len(files)))

        for file in files:
            if file[0] == ".":
                continue
            with open(root + "/" + file) as f:
                content = json.load(f)
                d = content["data"][0]

                data.append({**d["config"], **summarize(d["outcome"], d["config"])})

    return data

def create_df(data):
    df = pd.DataFrame(data)
    df.min_AB = df.min_AB.fillna(100)
    df.target = df.target.fillna("H/AB")

    return df


@click.command()
@click.option("--inp", help="directory with experiment files")
@click.option("--outp", help="output for dataframe", default="experiment_summary.csv")
def parse_data(inp, outp):
    print("Opening {}".format(inp))
    data = open_files(inp)

    df = create_df(data)
    print("Head of data:")
    print(df.head())

    print("Creating csv at {}".format(outp))
    df.to_csv(outp, index=False)

    print("All done")


if __name__ == "__main__":
    parse_data()

