# import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
import json
import os

def deployment():
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


if __name__ == "__main__":
  deployment()
  print("deployment returned")