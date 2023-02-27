import pandas as pd
import os


def jsonToDF(filePath: str) -> pd.DataFrame:
    """
    get json from file as dataframe
    """
    assert os.path.exists(filePath)
    df = pd.read_json(filePath, lines=True)

    return df
