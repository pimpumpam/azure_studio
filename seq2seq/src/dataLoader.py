import os
import glob
import yaml
import argparse
import pandas as pd 

from pathlib import Path
from functools import reduce
from datetime import datetime

from pyspark.sql.types import *
from pyspark.sql import DataFrame, SparkSession

from src.utils import find_item
# from utils import find_item

# Global variable
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
CONFIG_DIR = os.path.join(ROOT, 'seq2seq', 'configs')


def convert_dtype(data, dtype_dict):
    for col in data.columns:
        try:
            data = data.withColumn(col,
                                   data[col].cast(dtype_dict[col])
                                  )
        except:
            print(f"Could not convert dtype for \"{col}\" column")

    return data
    

class DataLoad():
    def __init__(self, framework='pyspark'):
        self.framework = framework

    def load_from_local(self, path, ext):
        data = []
        filelist = glob.glob(os.path.join(path, f"*.{ext}"))
        assert len(filelist)>0, f"Extension with {ext} not exist in Path({path})"
        
        if self.framework == 'pyspark':
            spark = SparkSession.builder \
                .config("spark.driver.memory", "15g") \
                .appName('suwon') \
                .getOrCreate()
            loader = getattr(getattr(spark, "read"), f"{ext}")
            
        elif self.framework == 'pandas':
            loader = getattr(pd, f"read_{ext}")
        
        for file in filelist:
            datum = loader(file)
            data.append(datum)
            
        if self.framework == 'pyspark':
            return reduce(DataFrame.unionAll, data)
            
        elif self.framework == 'pandas':
            data = pd.concat(data)
            # data = spark.createDataFrame(data)
            # print("[INFO] Convert Spark DF")
            return data
            
        
    def load_from_ndap(self, query, conn):
        """
        추후 개발 예정
        
        """

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='s1ap.yaml', help="데이터 config 파일명")
    args = parser.parse_args()

    # load configs            
    with open(os.path.join(CONFIG_DIR, 'data', args.data)) as y:
        DATA_CONFIG = yaml.load(y, Loader=yaml.FullLoader)
        
    # load data
    loader = DataLoad(framework=find_item(DATA_CONFIG, 'framework'))
    data = loader.load_from_local(path=find_item(DATA_CONFIG, 'path'), 
                                  ext=find_item(DATA_CONFIG, 'extension')
                                 )
    print("[INFO] Complete Load Data")
    
    # convert dtypes
    data = convert_dtype(data=data,
                         dtype_dict=find_item(DATA_CONFIG, 'dtype')
                        )
    print("[INFO] Complete Convert Data Type")

    print(data.head())
    print(data.dtypes)
