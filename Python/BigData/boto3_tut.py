
# http://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_use_switch-role-api.html

#NOTE: boto3 has a very diferent syntax compared to boto2. Refer to the page below to understand the differences
# http://boto3.readthedocs.org/en/latest/guide/migrations3.html

import boto3

### GET TEMP CREDENTIALS
#TODO: automate this task
# On the command line:
# python /Users/cka694/Resources/aws/awstokens.py
# Then get the credentials from vim ~/.aws/credentials

aws_access_key_id = 'ASIAITAPPL5VO5TYL4GQ'
aws_secret_access_key = 'UlZxmCsIfeKlURHW5oyqXLqvo3CfIgnJR5tpD2Mi'
aws_session_token = 'FQoDYXdzEDAaDDA0HmYAD0xiWSnafyKOAn5kfchGq0jZRIfLIDwo/dqhMKSgtORoB6QFuonspdn0h8TzTwnUlJNx9AUaSdqMkS5XL/gX/B420SKpTVsmGxKZKiCxys6sSRz94y3hTj9F6wSij2J/xmqTaX+KN9oUxKVK4yOaz4MnZG4SwL4NJnAoz6/A1WOlJEAoT7    Bx55egddpB0lll81+jfvsjcIqweWhFWJXN8t68yx1+GU0TiLcofh7hfGcQXBjqEVCq8afhTKm/9GOxYkXDdU2YhfaTWtwBlpiDWiUUaPC3T8svh9/COtpSINjImowuTTsczn2Uv/rP6GO6MfRgtbhXeJD1Xo1IMd6wS7IU9dDmnPTMctyzbXl0kNwohsFrwN3ENyjc/ei4BQ=='


###############################################################################
### CONNECT TO S3
# Use the temporary credentials that AssumeRole returns to make a
# connection to Amazon S3
s3_resource = boto3.resource(
    's3',
    aws_access_key_id = aws_access_key_id,
    aws_secret_access_key = aws_secret_access_key,
    aws_session_token = aws_session_token,
)

# Use the Amazon S3 resource object that is now configured with the
# credentials to access your S3 buckets.
for bucket in s3_resource.buckets.all():
    print(bucket.name)


### CREATE A BUCKET
s3_resource.create_bucket(Bucket='sbb-analytics-2')


### STORE DATA
s3_resource.Object('sbb-analytics-2', 'hello.txt').put(Body=open('/Users/cka694/Resources/aws/requirements.txt', 'rb'))


### ACCESS A BUCKET
import botocore
bucket = s3_resource.Bucket('sbb-analytics-2')
exists = True
try:
    s3_resource.meta.client.head_bucket(Bucket='sbb-analytics-2')
except botocore.exceptions.ClientError as e:
    # If a client error is thrown, then check that it was a 404 error.
    # If it was a 404 error, then the bucket does not exist.
    error_code = int(e.response['Error']['Code'])
    if error_code == 404:
        exists = False

for bucket in s3_resource.buckets.all():
    print(bucket.name)


### DELETE A BUCKET
# All of the keys in a bucket must be deleted before the bucket itself can be deleted
for key in bucket.objects.all():
    key.delete()
bucket.delete()


### ITERATE THROUGH BUCKETS AND KEYS
# Bucket and key objects provide collection attributes which can be iterated
for bucket in s3_resource.buckets.all():
    for key in bucket.objects.all():
        print(key.key)


### GRANT AND VIEW ACCESS CONTROLS
# Getting and setting canned access control values in Boto 3 operates on an ACL resource object:
bucket.Acl().put(ACL='public-read')

# It's also possible to retrieve the policy grant information:
acl = bucket.Acl()
for grant in acl.grants:
    print(grant['Grantee']['DisplayName'], grant['Permission'])

# To grant access to a user (you can also do this on AWS console)
bucket.Acl.put(GrantRead='emailAddress=user@domain.tld')


### GET KEY META DATA
key.put(Metadata={'meta1': 'This is my metadata value'})
print(key.metadata['meta1'])


###############################################################################
bucket = s3_resource.Bucket('sbb-analytics')

for key in bucket.objects.all():
    print(key.key)

bucket.objects('hello.txt').get()


###############################################################################
############################## USE BOTO S3 CLIENT #############################
###############################################################################

import boto3
s3_client = boto3.client(
    's3',
    aws_access_key_id = aws_access_key_id,
    aws_secret_access_key = aws_secret_access_key,
    aws_session_token = aws_session_token,
)

s3_client.download_file('sbb-analytics', 'historicalBalances.json.gz', '/Users/cka694/Documents/historicalBalances.json.gz')
s3_client.download_file('sbb-analytics', 'institution.json.gz', '/Users/cka694/Documents/institution.json.gz')
s3_client.download_file('sbb-analytics', 'profile.json.gz', '/Users/cka694/Documents/profile.json.gz')
s3_client.download_file('sbb-analytics', 'transactionCategory.json.gz', '/Users/cka694/Documents/transactionCategory.json.gz')
s3_client.download_file('sbb-analytics', 'transactions.json.gz', '/Users/cka694/Documents/transactions.json.gz')



# Upload the file to S3
open('/Users/cka694/Resources/aws/hello-client.txt').write('Hello, world!')
s3_client.upload_file('/Users/cka694/Resources/aws/requirements.txt', 'sbb-analytics', 'requirements-test.txt')

# Download the file from S3
s3_client.download_file('sbb-analytics', 'requirements-test.txt', '/Users/cka694/Documents/requirements-test.txt')

s3_client.download_file('sbb-analytics', 'historicalBalances.json.gz', '/Users/cka694/Documents/historicalBalances.json.gz')

s3_client.download_file('sbb-analytics', 'test-encryption.txt', '/Users/cka694/Documents/test.txt')
print(open('/Users/cka694/Resources/aws/hello2.txt').read())







# TO ADD EVERYONE
"""
,
		{
			"Sid": "Allow access to bucket",
			"Effect": "Allow",
			"Principal": {
				"AWS": "*"
			},
			"Action": "s3:*",
			"Resource": [
				"arn:aws:s3:::sbb-analytics/*",
				"arn:aws:s3:::sbb-analytics"
			]
		}
"""