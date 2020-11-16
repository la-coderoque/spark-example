import argparse
import os
from datetime import datetime
import calendar

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.functions import col, udf

spark = SparkSession.builder.appName('hdd').getOrCreate()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', type=str, help='Input dir for csv files')
    parser.add_argument('outdir', type=str, help='Output dir for csv file')
    args = parser.parse_args()
    
    if not os.path.exists(args.indir):
        raise OSError(f'Input path does not exist: {args.indir}')
    if os.path.exists(args.outdir):
        raise OSError(f'Output path already exist: {args.outdir}')
    
    return args.indir, args.outdir


def b_to_gb(size):
    """ Convert Bytes to Gigabytes
    """
    if int(size) <= 0:
        return size
    return str(round(int(size) / 10**9))

b_to_gb_UDF = udf(lambda z: b_to_gb(z), StringType())


def date_to_week_day(str_date):
    """ Convert date to weekday
    """
    if not str_date:
        return str_date
    return calendar.day_name[datetime.strptime(str_date, '%Y-%m-%d').weekday()]

date_to_week_day_UDF = udf(lambda z: date_to_week_day(z), StringType())


def main(indir, outdir):
    # read
    df = spark.read.csv(indir, header='true')
    df = df.select(
        'date', 'serial_number', 'model', 'capacity_bytes', 'failure'
    )
    df = df.fillna({'capacity_bytes': '0'})
    df = df.withColumn('failure', col('failure').cast(IntegerType()))

    # bytes
    df = df.withColumn('capacity_GB', b_to_gb_UDF(col('capacity_bytes')))
    df = df.drop('capacity_bytes')
    df = df.withColumn('capacity_GB', df['capacity_GB'].cast(IntegerType()))

    # days
    df = df.withColumn('weekday', date_to_week_day_UDF(col('date')))
    df = df.drop('date')

    # df with count of unique hdds by serial numbers
    unique_df = df.select('serial_number', 'model', 'capacity_GB')
    unique_df = unique_df.distinct()
    unique_df.drop('serial_number')
    unique_df = unique_df.groupBy('model', 'capacity_GB').count()

    # sum of hdds failures by models
    df = df.drop('serial_number')
    failure_df = df.groupBy('model', 'capacity_GB').sum('failure')

    # df with failures probabilities
    probability_df = unique_df.join(
        failure_df, on=['model', 'capacity_GB']
    ).withColumn(
        'failure_probability',
        (col('sum(failure)') / col('count')),
    ).drop(
        'count', 'sum(failure)',
    )

    # pivot by weekdays
    df = df.join(probability_df, on=['model', 'capacity_GB'])
    df = df.groupBy(
        'model', 'capacity_GB', 'failure_probability'
    ).pivot('weekday').sum('failure')

    # write to csv
    df.coalesce(1).write.option('header', 'true').csv(outdir)


if __name__ == '__main__': 
    main(*parse_args())