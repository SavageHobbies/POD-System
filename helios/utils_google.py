from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import gspread
from google.oauth2.service_account import Credentials


def get_gspread_client(service_json: str | None) -> gspread.Client | None:
    if not service_json:
        return None
    try:
        # Expand tilde in path if present
        expanded_path = os.path.expanduser(service_json)
        if os.path.exists(expanded_path):
            info = json.loads(open(expanded_path, "r").read())
        else:
            info = json.loads(service_json)
    except Exception:
        return None
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)


def append_rows(sheet_id: str, worksheet_name: str, rows: List[List[Any]], service_json: str | None) -> bool:
    client = get_gspread_client(service_json)
    if not client:
        return False
    sh = client.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=100, cols=30)
    ws.append_rows(rows, value_input_option="USER_ENTERED")
    return True
