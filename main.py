import os
import tempfile

from google.cloud import storage
storage_client = storage.Client()

def speech2emotionAlgo(data, context):
    #Download file to cloud local
    #File local path: temp_path
    file_data = data
    
    file_name = file_data['name']
    bucket_name = file_data['bucket']
    _, temp_local_filename = tempfile.mkstemp()
    
    blob = storage_client.bucket(bucket_name).get_blob(file_name)
    blob.download_to_filename(temp_local_filename)
    
    print(f'Audio {file_name} was downloaded to {temp_local_filename}.')







    #Update file's metadata
    #'Emotion Flag' is the speech2emotion algorithm return result
    #Upload blob back to same filename
    metadata = {'EmotionFlag': 'xxxxxx'}
    blob.metadata = metadata
    blob.patch()
    blob.upload_from_filename(file_name)



    #Delete the cloud local temp file
    os.remove(temp_local_filename)
    return

