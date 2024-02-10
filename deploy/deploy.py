# import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account


def deployment():
  creds = service_account.Credentials.from_service_account_file('../../secret.json')

  try:
    service = build("drive", "v3", credentials=creds)
    file_metadata = {
      "name": "main_upstream.ipynb",
      'parents':['parent_folder_id']
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