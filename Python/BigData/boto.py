###############################################################################
__author__ = 'amirkavousian'
__email__ = 'amir.kavousian@sunrun.com'
# created on: Aug 26, 2015
# summary: boto playground
# Other resources:
# http://boto.readthedocs.org/en/latest/s3_tut.html
###############################################################################

########################################
### LOAD STANDARD MODULES

from BigData.boto.s3.connection import S3Connection
from BigData.boto.s3.key import Key

### Connect to S3 bucket
secretKey = 'AKIAIXPL4ZWH6PCFTPGA'
awsKey = 'v9p9FGAy3JPffFpKTXun99gEZhOh1hU14Jx4TDWF'
bucketName = 'environmental-data'
conn = S3Connection(secretKey, awsKey)
buck = conn.get_bucket(bucketName)
########################################

########################################
# Find S3 availability regions
from BigData.boto.s3.connection import Location
print '\n'.join(i for i in dir(Location) if i[0].isupper())
########################################


########################################
### Storing and retrieving strings
# To store new data in S3, start by creating a new Key object
# key is the equivalent of the file name.
# S3 does not have a hierarchical directory structure. Once a bucket is created, any number of files can be stored under it.
# To synthesize a hierarchy in the bucket, use the '/' character. Interfaces such as CyberDuck recognize this as the directory structure.
# The first thing to create or access an individual file is to generate an instance of a Key object.
# A Key instance is a pointer to a file on S3. By setting the .key attribute of key instance, we can point it to specific files on S3.
k = Key(buck)  # this is just an instance. Change its key attribute to point to specific files.
k.key = 'test/foobar'  # creates a directory called 'test' and a file called 'foobar' under it. The file contents will be set separately.
k.set_contents_from_string('This is a test of S3 - woohoo')
# The net effect of these statements is to create a new object in S3 with a key of “foobar” and a value of “This is a test of S3”.
# You can change the key and assign new content to it. k is one instance of Key objects within the bucket.
# But by changing its key value, you can assign different values to it.

# To retrieve the file contents as string:
k = Key(buck)
k.key = 'test/foobar'
k.get_contents_as_string()


### Storing and retrieving local files
k = Key(buck)
k.key = 'test/image1'  # the name of the file as stored on S3
# upload the file
k.set_contents_from_filename('/Users/amirkavousian/Documents/PROJECTS/EnvironmentalData/image1.jpeg')

# To retrieve the contents as another local file
k.get_contents_to_filename('/Users/amirkavousian/Documents/PROJECTS/EnvironmentalData/image1-returned3.jpg')
# Note: boto streams the content to and from S3 so you should be able to send and receive large files without any problem.


### When fetching a key that already exists, you have two options.
# (a) If you’re uncertain whether a key exists (or if you need the metadata set on it, you can call Bucket.get_key(key_name_here).
# will hit the API
possible_key = buck.get_key('foobar')
# will not hit the API
key_we_know_is_there = buck.get_key('foobar', validate=False)

# (b) However, if you’re sure a key already exists within a bucket, you can skip the check for a key on the server.
for bl in buck.list():
    print bl
#######################################


#######################################
### ACCESSING AND EXPLORING A BUCKET
# to check the existence of a bucket
# get an iterator object of the keys in the bucket
buck.list()
for bl in buck.list():
    print bl

# get a list of all keys in the bucket
existing_keys = buck.get_all_keys(maxkeys=10)
type(existing_keys[0])

# Check the exitence of a bucket before sending queries to it
# If the bucket does not exist, a S3ResponseError will commonly be thrown. If you’d rather not deal with any exceptions, you can use the lookup method.
existent = conn.lookup('environmental-data')
nonexistent = conn.lookup('i-dont-exist-at-all')
if nonexistent is None:
    print "No such bucket!"

# list all buckets
bucks = conn.get_all_buckets()
len(bucks)
for b in bucks:
    print b.name
#######################################


#######################################
### Storing large data files
# S3 allows you to split such files into smaller components.
import math, os
from filechunkio import FileChunkIO

# get file info
source_path = '/Users/amirkavousian/Documents/PROJECTS/EnvironmentalData/image1.jpeg'
source_size = os.stat(source_path).st_size

# Create a multipart upload request
mp = buck.initiate_multipart_upload(os.path.basename(source_path))

# Use a chunk size of 50 MiB (feel free to change this)
chunk_size = 52428800
chunk_count = int(math.ceil(source_size / float(chunk_size)))

# Send the file parts, using FileChunkIO to create a file-like object
# that points to a certain byte range within the original file. We
# set bytes to never exceed the original file size.
for i in range(chunk_count):
    offset = chunk_size * i
    bytes = min(chunk_size, source_size - offset)  # taking care of the last chunk, when the chunk size will be smaller than the chunk_size
    with FileChunkIO(source_path, 'r', offset=offset, bytes=bytes) as fp:
        mp.upload_part_from_file(fp, part_num=i + 1)

# Finish the upload
mp.complete_upload()


### Multi-thread upload
# It is also possible to upload the parts in parallel using threads. The s3put script that ships with Boto
buck.get_all_multipart_uploads()

# More on downloading files larger than 5GB:
# http://stackoverflow.com/questions/28116881/amazon-s3-upload-fails-using-boto-python

#######################################


#######################################
### Setting Access Control List (ACL)
# (a) private: Owner gets FULL_CONTROL. No one else has any access rights.
# (b) public-read: Owners gets FULL_CONTROL and the anonymous principal is granted READ access.
# (c) public-read-write: Owner gets FULL_CONTROL and the anonymous principal is granted READ and WRITE access.
# (d) authenticated-read: Owner gets FULL_CONTROL and any principal authenticated as a registered Amazon S3 user is granted READ access.

# ACL can be set at the bucket or key level.
# For example, to make a bucket readable by anyone
buck.set_acl('public-read')
# To set the ACL for Key objects
buck.set_acl('public-read', 'foobar')
# Or, use the key object
k.set_acl('public-read')

# Get the current ACL for a bucket or key
acp = b.get_acl()
acp
acp.acl
acp.acl.grants
for grant in acp.acl.grants:
  print grant.permission, grant.display_name, grant.email_address, grant.id
#######################################


#######################################
### Setting metadata on keys
k = Key(buck)
k.key = 'has_metadata2'
k.set_metadata('meta1', 'This is the first metadata value')
k.set_metadata('meta2', 'This is the second metadata value')
# k.set_contents_from_filename('/Users/amirkavousian/Documents/PROJECTS/EnvironmentalData/image1.jpeg')
k.set_contents_from_string('/Users/amirkavousian/Documents/PROJECTS/EnvironmentalData/image1.jpeg')
k.get_contents_as_string()

# To retrieve the meta data values
k = buck.get_key('has_metadata')
k.get_metadata('meta1')
k.get_metadata('meta2')
#######################################


#######################################
### Lifecycle policy
from BigData.boto.s3.lifecycle import Lifecycle, Transition, Rule

# Configure a lifecycle policy (not yet assigned to any specific bucket)
to_glacier = Transition(days=30, storage_class='GLACIER')
rule = Rule('ruleid', 'logs/', 'Enabled', transition=to_glacier)
lifecycle = Lifecycle()
lifecycle.append(rule)

# Set the lifecycle policy for the bucket
buck.configure_lifecycle(lifecycle)

# Get the current lifecycle policy for the bucket
current = buck.get_lifecycle_config()
print current[0].transition


### Restore objects from Glacier
# It takes about 4 hours for a restore operation to make a copy of the archive available for you to access.
key = buck.get_key('foobar')
key.restore(days=5)  # number of days to keep in S3 before sending back to Glacier

# To see whether a file is stored in S3 or Glacier
for key in buck.list():
  print key, key.storage_class
#######################################
