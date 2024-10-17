import os
import boto3

from dotenv import load_dotenv
from os.path import join, dirname
from serpapi import GoogleSearch

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

def upload_image_to_s3(file_path, bucket_name, image_name, acl="public-read"):
    # Create an S3 client
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    
    # Upload the file to the specified S3 bucket
    with open(file_path, "rb") as f:
        s3.upload_fileobj(f, bucket_name, image_name, ExtraArgs={"ACL": acl})

    # Get the region name
    bucket_location = s3.get_bucket_location(Bucket=bucket_name)
    region_name = bucket_location['LocationConstraint']

    # Construct the image URL
    if region_name:
        image_url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{image_name}"
    else:
        image_url = f"https://{bucket_name}.s3.amazonaws.com/{image_name}"

    return image_url

def search_title(image_url):
    # Parameters for the search
    params = {
        "engine": "google_reverse_image",
        "image_url": image_url,
        "api_key": os.getenv("SERPAPI_API_KEY"),
    }
    # Perform the search using SerpAPI
    search = GoogleSearch(params)
    results = search.get_dict()
    # Extract the title from the first image result
    inline_images = results['image_results'][0]['title']
    
    return inline_images