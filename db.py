import seaborn as sns
from src.utils.utils import data_from_db,insert_data_db
from dotenv import load_dotenv
import os
load_dotenv()
url=os.getenv('url')
db=os.getenv('database')
collection=os.getenv('collection2')

df=sns.load_dataset('iris')
insert_data_db(db,collection,df)