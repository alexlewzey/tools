"""Google Cloud Platform Tool-Kit.

NOTE: set NUMERIC columns to floats64 in large queries pd.read_gbq will read NUMERIC
columns as strings (take up loads of memory)

Sql query to DataFrame
----------------------
import pandas as pd
from google.cloud import bigquery

bq_client = bigquery.Client()
query = 'SELECT * FROM dataset.table'
df = pd.read_gbq(query, use_bqstorage_api=True, progress_bar_type='tqdm')


push DataFrame to bigquery
--------------------------
bq_client = bigquery.Client()
pred.to_gbq(destination_table='dataset.table_name',
            project_id=os.environ['GCLOUD_PROJECT'],
            if_exists='replace',
            progress_bar=True,
            location='europe-west2')

when pushing large DataFrames to bigquery (1GB+) use gcptk.df2gcs2bq()


Read/write DataFrame directly from/to GCS bucket
------------------------------------------------
client = storage.Client()
df = pd.read_csv('gs://bucket/path/to/file.csv')
df.to_csv('gs://bucket/path/to/file.csv')
"""

import logging
import re

import pandas as pd
from google.cloud import bigquery, storage
from tqdm import tqdm

logger = logging.getLogger(__name__)


def df2gcs2bq(
    df: pd.DataFrame,
    destination_table: str,
    bucket: str,
    temporary_blob_dir: str,
    append: bool = True,
    chunksize: int = 500_000,
) -> None:
    """Load a large DataFrame to bigquery by first staging it in gcs as chunks in
    parquet format before loading to bigquery. The parquet files are automatically
    removed from gcs after loading.

    Args:
        df: DataFrame to be loaded
        destination_table: bigquery upload destination eg dataset.table
        bucket: gcs bucket where data will be staged before being loaded to bigquery,
            defaults to environment variable BUCKET
        temporary_blob_dir: temporary staging directory within gcs
        append: boolean True=append existing table False=overwrite existing table
        chunksize: no. of rows in the individual parquet files

    Returns:
        None

    example
    -------
    gcptk.df2gcs2bq(df, 'dataset.table', append=False)
    """
    dataset, table = destination_table.split(".")
    # save DataFrame as parquet chunks in gcs
    idxs = list(range(0, df.shape[0], chunksize))
    temporary_blob_dir = temporary_blob_dir.rstrip("/")
    for i in tqdm(idxs):
        uri = f"gs://{bucket}/{temporary_blob_dir}/chunk{i}.parquet"
        df.iloc[i : i + chunksize, :].to_parquet(uri)
    # move data from gcs to bq
    client_bq = bigquery.Client()
    dataset_ref = client_bq.get_dataset(dataset)
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.PARQUET
    job_config.write_disposition = (
        bigquery.WriteDisposition.WRITE_APPEND
        if append
        else bigquery.WriteDisposition.WRITE_TRUNCATE
    )
    load_uri = f"gs://{bucket}/{temporary_blob_dir}/chunk*.parquet"
    load_job = client_bq.load_table_from_uri(
        load_uri, dataset_ref.table(table), job_config=job_config
    )
    logger.info(f"Starting job {load_job.job_id}")
    load_job.result()  # Waits for table load to complete.
    # removing blobs from gcs
    client_gcs = storage.Client()
    bucket = client_gcs.get_bucket(bucket)
    blobs = bucket.list_blobs(prefix=f"{temporary_blob_dir}/")
    for blob in tqdm(blobs):
        # check file is one of the chunks created in staging before deleting
        if re.search(r".*chunk\d*\.parquet", blob.path):
            logger.info(f"removing: {blob.path}")
            blob.delete()


def to_gbq(
    df: pd.DataFrame,
    destination_table: str,
    if_exists: str = "replace",
    project_id: str | None = None,
    **kwargs,
) -> None:
    """Wrapper for pushing DataFrames to bigquery, note duplicate col names are not
    accepted replace or append."""
    client = bigquery.Client()
    project_id = project_id if project_id else client.project
    df.to_gbq(
        destination_table=destination_table,
        project_id=project_id,
        if_exists=if_exists,
        location="europe-west2",
        **kwargs,
    )


def read_gbq(query: str, use_bqstorage_api=True, **kwargs) -> pd.DataFrame:
    """Convenience wrapper around pd.read_gbq."""
    return pd.read_gbq(
        query, use_bqstorage_api=use_bqstorage_api, progress_bar_type="tqdm", **kwargs
    )
