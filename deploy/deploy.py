# import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2 import service_account
import json
import os
import io

def from_github_to_colab():
  creds = service_account.Credentials.from_service_account_info(json.loads(os.getenv('DRIVE_CONFIG')))

  try:
    service = build("drive", "v3", credentials=creds)
    file_metadata = {
      "name": "main_upstream.ipynb",
      'parents':[os.getenv('PARENT_FOLDER_ID')]
    }
    media = MediaFileUpload("../main.ipynb")
    file = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id")
        .execute()
    )
    print(f'File ID: {file.get("id")}')

  except HttpError as error:
    print(f"An error occurred: {error}")
    file = None

  return file.get("id")

def push_colab_to_github(real_file_id):

  creds = service_account.Credentials.from_service_account_info(json.loads(os.getenv('DRIVE_CONFIG')))

  try:
    service = build("drive", "v3", credentials=creds)
    file_id = real_file_id
    request = service.files().get_media(fileId=file_id)
    file = io.BytesIO()
    downloader = MediaIoBaseDownload(file, request)
    done = False
    while done is False:
      status, done = downloader.next_chunk()
      print(f"Download {int(status.progress() * 100)}.")

  except HttpError as error:
    print(f"An error occurred: {error}")
    file = None

  return file.getvalue()


if __name__ == "__main__":
  # Enable flags for future
  # from_github_to_colab()
  # files = {"main.ipynb" : "file_id"}
  # for file_name, file_id in files.items(): 
  #   file_content = push_colab_to_github(real_file_id=file_id)
  #   with open(f"../src/{file_name}", 'wb') as f_out:
  #     f_out.write(file_content)
  print("deployment returned")