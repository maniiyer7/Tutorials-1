__author__ = 'cstaab'

import csv
import hashlib
import hmac
import base64
import urlparse
import requests
import os
from ratelimiting import RateLimiting


def sign_url(input_url=None, client_id=None, client_secret=None):
  """ Sign a request URL with a Crypto Key.

      Usage:
      from urlsigner import sign_url

      signed_url = sign_url(input_url=my_url,
                            client_id=CLIENT_ID,
                            client_secret=CLIENT_SECRET)

      Args:
      input_url - The URL to sign
      client_id - Your Client ID
      client_secret - Your Crypto Key

      Returns:
      The signed request URL
  """

  # Return if any parameters aren't given
  if not input_url or not client_id or not client_secret:
    return None

  # Add the Client ID to the URL
  input_url += "&client=%s" % (client_id)

  url = urlparse.urlparse(input_url)

  # We only need to sign the path+query part of the string
  url_to_sign = url.path + "?" + url.query

  # Decode the private key into its binary format
  # We need to decode the URL-encoded private key
  decoded_key = base64.urlsafe_b64decode(client_secret)

  # Create a signature using the private key and the URL-encoded
  # string using HMAC SHA1. This signature will be binary.
  signature = hmac.new(decoded_key, url_to_sign, hashlib.sha1)

  # Encode the binary signature into base64 for use within a URL
  encoded_signature = base64.urlsafe_b64encode(signature.digest())

  original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

  # Return signed URL
  return original_url + "&signature=" + encoded_signature

rate_limiter = RateLimiting(max_calls=10, period=1.0)

baseurl = 'https://maps.googleapis.com/maps/api/geocode/json?address='

i = 0
with open(os.path.dirname(os.path.realpath(__file__)) + '\\Penetration analysis standardization\\Combined Sunrun-permit data 15July15 test set.csv','rb') as infile:
    reader = csv.DictReader(infile)
    with open(os.path.dirname(os.path.realpath(__file__)) + '\\Penetration analysis standardization\\Combined Sunrun-permit data 15July15 batch 1 - standardized.csv','wb') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Owner first name', 'Owner last name', 'Address', 'City', 'State', 'Zip code', 'System builder',
                         'System cost', 'Permit issue date', 'Permit number', 'Standardized address', 'Standardized City',
                         'Standardized state', 'Standardized zip code'])
        with rate_limiter:
            for row in reader:
                address = (row['Address'] + ' ' + row['City'] + ' ' + row['State'] + ' ' + row['Zip code']).replace(' ','+')
                print address
                url = baseurl + address
                r = requests.get(sign_url(url,'gme-sunruninc1','SnSXaevui20LZlHuXtS9GHIhQZU=')).json()

                try:
                    for entry in r['results'][0]['address_components']:
                        if 'street_number' in entry['types']:
                            street_number = entry['short_name'].encode('utf-8')
                        elif 'route' in entry['types']:
                            street = entry['short_name'].encode('utf-8').upper()
                        elif 'locality' in entry['types']:
                            city = entry['short_name'].encode('utf-8').upper()
                        elif 'administrative_area_level_1' in entry['types']:
                            state = entry['short_name'].encode('utf-8').upper()
                        elif 'postal_code' in entry['types']:
                            zip_code = entry['short_name'].encode('utf-8')
                    place_id = r['results'][0]['place_id']
                    print street_number, street, city, state, zip_code, place_id

                    print r
                except:
                    print r
                    '''
                    for entry in r['results'][0]['address_components']:
                        if 'street_number' in entry['types']:
                            street_number = entry['short_name'].encode('utf-8')
                        elif 'route' in entry['types']:
                            street = entry['short_name'].encode('utf-8').upper()
                        elif 'locality' in entry['types']:
                            city = entry['short_name'].encode('utf-8').upper()
                        elif 'administrative_area_level_1' in entry['types']:
                            state = entry['short_name'].encode('utf-8').upper()
                        elif 'postal_code' in entry['types']:
                            zip_code = entry['short_name'].encode('utf-8')
                        else:
                            pass
                    address_uid = str(street_number) + ' ' + street + city + state + zip_code
                    writer.writerow([row['Contract Name'], row['Home Address'], row['City'], row['State'], row['Zip'],
                                     row['Concatenated address'],street_number,street,city,state,zip_code,address_uid])
                    print str(i) + " rows complete"
                    i += 1
                except:
                    i += 1
            else:
                i += 1
                '''
