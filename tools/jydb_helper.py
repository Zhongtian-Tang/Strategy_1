import pandas as pd
import numpy as np
import datetime
import logging
from sqlalchemy import create_engine


connection_str = 'oracle+cx_oracle://rejy:jcFXLzL10@10.224.6.3:1522/?service_name=orcl'
engine = create_engine(connection_str)
conn = engine.connect()


class Da