import argparse
import io
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import yaml


SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")  # 입력 없을 시, 기본값으로 config.yaml을 가져옴
parser = parser.parse_args()
with open(os.path.join("../config", parser.config)) as f:
    CFG = yaml.safe_load(f)
TOKEN = CFG["gdrive"]["token"]
CREDENTIALS = CFG["gdrive"]["credentials"]
FOLDER_ID = CFG["gdrive"]["folder_id"]


class GoogleDriveManager:
    def __init__(self):
        self.service = self.get_drive_service()
        self.root_folder_id = FOLDER_ID

    def get_drive_service(self):
        creds = None
        if os.path.exists(TOKEN):
            creds = Credentials.from_authorized_user_file(TOKEN, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS, SCOPES)
                creds = flow.run_local_server(port=0)

            with open(TOKEN, "w") as token:
                token.write(creds.to_json())

        return build("drive", "v3", credentials=creds)

    def find_folder_id_by_name(self, folder_name, parent_folder_id=None):
        """폴더명으로 폴더 ID 찾기"""
        if not parent_folder_id:
            parent_folder_id = self.root_folder_id

        # 특정 폴더명과 정확히 일치하는 폴더 검색 쿼리
        query = f"name='{folder_name}' and "
        query += f"'{parent_folder_id}' in parents and "
        query += "mimeType='application/vnd.google-apps.folder' and "
        query += "trashed=false"

        try:
            results = (
                self.service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="files(id, name)",
                    pageSize=1,  # 첫 번째 일치하는 폴더만 필요
                )
                .execute()
            )

            files = results.get("files", [])

            if not files:
                print(f"폴더를 찾을 수 없습니다: {folder_name}")
                return None

            return files[0]["id"]

        except Exception as e:
            print(f"폴더 검색 중 오류 발생: {str(e)}")
            return None

    def upload_json_data(self, json_string, filename, folder_id=None):
        """직렬화된 JSON string 직접 업로드"""
        try:
            # 메모리 스트림으로 변환
            file_stream = io.BytesIO(json_string.encode("utf-8"))

            # 파일 메타데이터 설정
            file_metadata = {"name": filename, "mimeType": "application/json"}
            if folder_id:
                file_metadata["parents"] = [folder_id]

            # 미디어 객체 생성
            media = MediaIoBaseUpload(file_stream, mimetype="application/json", resumable=True)

            # 파일 업로드
            file = self.service.files().create(body=file_metadata, media_body=media, fields="id, name").execute()
            return file

        except Exception as e:
            print(f"Error uploading JSON data: {str(e)}")
            return None

    def upload_dataframe(self, dataframe, filename, folder_id=None):
        """Pandas DataFrame 직접 업로드"""
        try:
            # DataFrame을 CSV 스트림으로 변환
            buffer = io.StringIO()
            dataframe.to_csv(buffer, index=False)
            file_stream = io.BytesIO(buffer.getvalue().encode("utf-8"))

            # 파일 메타데이터 설정
            file_metadata = {"name": filename, "mimeType": "text/csv"}
            if folder_id:
                file_metadata["parents"] = [folder_id]

            # 미디어 객체 생성
            media = MediaIoBaseUpload(file_stream, mimetype="text/csv", resumable=True)

            # 파일 업로드
            file = self.service.files().create(body=file_metadata, media_body=media, fields="id, name").execute()
            return file

        except Exception as e:
            print(f"Error uploading DataFrame: {str(e)}")
            return None


if __name__ == "__main__":
    drive_manager = GoogleDriveManager()
    # 파일 목록 조회
    files = drive_manager.list_folder_files()
    for file in files:
        print(f"Name: {file['name']}, ID: {file['id']}")
