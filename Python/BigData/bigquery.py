###############################################################################
__author__ = 'amirkavousian'
__email__ = 'amir.kavousian@sunrun.com'
# created on: October 06, 2015
# summary: exploration of Python's Google BigQuery analysis tools
###############################################################################


import googleapiclient as gac


### NYC TAXI DATA FROM Google BigQuery
# http://minimaxir.com/2015/08/nyc-map/
# http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml
# http://stackoverflow.com/questions/13212991/how-to-query-bigquery-programmatically-from-python-without-end-user-interaction

# Query from BigQuery
query = """
    SELECT ROUND(pickup_latitude, 4) as lat,
    ROUND(pickup_longitude, 4) as long,
    COUNT(*) as num_pickups
    FROM [nyc-tlc:yellow.trips_2014]
    GROUP BY lat, long
    """


import httplib2
from apiclient.discovery import build
from oauth2client.client import SignedJwtAssertionCredentials
import crypto


### PROJECT CONFIGS
proj_name = 'maps'
proj_id = 'focused-history-109102'
proj_num = '506319953602'
# service account email
google_api_email = 'amir.kavousian@sunrunhome.com'
key = 'AIzaSyDlTboED86C-GX4nTVd3MEx6HE21VGosyY'


#######################################
### CONNECT VIA API
# OBTAIN THE KEY FROM THE GOOGLE APIs CONSOLE
# More instructions here: http://goo.gl/w0YA0
f = file('key.p12', 'rb')
key = f.read()
f.close()


# Build the credentials
credentials = SignedJwtAssertionCredentials(
    google_api_email,
    key,
    scope='https://www.googleapis.com/auth/bigquery')


### CALL BigQuery
http = httplib2.Http()
http = credentials.authorize(http)

service = build('bigquery', 'v2')
datasets = service.datasets()
response = datasets.list(projectId=proj_num).execute(http)

print 'Dataset list:'
for dataset in response['datasets']:
  print '%s' % dataset['datasetReference']['datasetId']
#######################################

#######################################
### USING pandas IO tools
# http://pandas.pydata.org/pandas-docs/stable/io.html#io-bigquery
#######################################

#######################################
### CONNECT VIA PYTHON
# http://stackoverflow.com/questions/31590404/bigquery-in-python-how-to-put-the-results-of-a-query-in-a-table
query_data = {'configuration': {
        'query': {
            'query': QUERY
            'destinationTable': {
                'projectId': project_id,
                'datasetId': dataset_id,
                'tableId': 'table_id'
            },
            'createDisposition': 'CREATE_IF_NEEDED',
            'writeDisposition': 'WRITE_TRUNCATE',
            'allowLargeResults': True
        },
        }
    }
query_request.query(projectId=PROJECT_NUMBER,body=query_data).execute()



# Another stackoverflow
job_data = {
        'jobReference': {
            'projectId': myProjectId,
            'jobId': str(uuid.uuid4())
        },
        'configuration': {
            'extract': {
                'sourceTable': {
                    'projectId': sharedProjectId,
                    'datasetId': sharedDatasetId,
                    'tableId': sharedTableId,
                },
                'destinationUris': [cloud_storage_path],
                'destinationFormat': 'AVRO'
            }
        }
    }

service.jobs().insert(projectId=myProjectId, body=job_data).execute()
#######################################

#######################################
# https://www.linkedin.com/pulse/use-python-load-local-files-google-cloud-storage-charles-clifford
import sys
import httplib2
import os
import random
import time
import traceback
import re
import logging
from apiclient import discovery
from oauth2client.client import GoogleCredentials
from apiclient.errors import HttpError
from apiclient.http import MediaFileUpload
from json import dumps as json_dumps

credentials = GoogleCredentials.get_application_default()

service = discovery.build('storage', 'v1', credentials=credentials)

directory_name = argv[1]

if os.path.exists(directory_name):
    for file_name in os.listdir(directory_name):

    local_file=os.path.join('/',directory_name, file_name)
        if os.path.isfile(local_file):
            justa_file_name = os.path.splitext(file_name)[0]
            search_result= re.search('\d{8}', justa_file_name, flags=0)
            if search_result: # the pattern has been found within file name
            #remove _CCYYMMDD from file name
            justa_file_name=justa_file_name[:-9]

            fields_to_return = 'nextPageToken,items(name,size,contentType,metadata(my-key))'
            req = service.objects().list(bucket=justa_file_name, fields=fields_to_return)

            while req is not None:
                resp = req.execute()
                items = resp["items"]
                for item in items:
                if item["name"] == file_name:
                # delete existing file
                service.objects().delete(bucket=bucket_name, object=file_name ).execute()
                logging.info('Deleted existing file: %s from bucket %s'%(file_name,bucket_name))
                req = service.objects().list_next(req, resp)

                logging.info('Building upload request...')
                media = MediaFileUpload(local_file, chunksize=CHUNKSIZE, resumable=True)
                if not media.mimetype():
                media = MediaFileUpload(local_file, DEFAULT_MIMETYPE, resumable=True)
                request = service.objects().insert(bucket=bucket_name, name=blob_name,
                media_body=media)

                logging.info('Uploading file: %s, to bucket: %s, blob: %s ' % (local_file, bucket_name,
                blob_name))

                progressless_iters = 0
                response = None
                while response is None:
                error = None
                try:
                progress, response = request.next_chunk()
                if progress:
                logging.info('Upload %d%%' % (100 * progress.progress()))
                except HttpError, err:
                error = err
                if err.resp.status < 500:
                raise
                except RETRYABLE_ERRORS, err:
                error = err

                if error:
                progressless_iters += 1
                handle_progressless_iter(error, progressless_iters)
                else:
                progressless_iters = 0

                logging.info('Upload complete!')

                logging.info('Uploaded Object:')

                logging.info(json_dumps(response, indent=2))

                The static variables, and method, the above statements depend upon are shown here (and are purely Google's creation):

                # Retry transport and file IO errors.
                RETRYABLE_ERRORS = (httplib2.HttpLib2Error, IOError)

                # Number of times to retry failed downloads.
                NUM_RETRIES = 5


# Number of bytes to send/receive in each request.
CHUNKSIZE = 2 * 1024 * 1024

# Mimetype to use if one can't be guessed from the file extension.
DEFAULT_MIMETYPE = 'application/octet-stream'


def handle_progressless_iter(error, progressless_iters):
if progressless_iters > NUM_RETRIES:
logging.info('Failed to make progress for too many consecutive iterations.')
raise error

sleeptime = random.random() * (2**progressless_iters)
logging.info('Caught exception (%s). Sleeping for %s seconds before retry #%d.'
% (str(error), sleeptime, progressless_iters))

time.sleep(sleeptime)
#######################################
