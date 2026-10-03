from enum import Enum
import os
from workspace import config
import pandas as pd
from pathlib import Path

class Gender(Enum):
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"

class User:
    def __init__(self, first_name="", last_name="", age=0, gender=Gender.OTHER):
        self._user_id = 0
        self._first_name = first_name
        self._last_name = last_name
        self._age = age
        self._gender = gender

    @property
    def first_name(self) -> str:
        return self._first_name

    @first_name.setter
    def first_name(self, value: str) -> None:
        self._first_name = value

    @property
    def last_name(self) -> str:
        return self._last_name

    @last_name.setter
    def last_name(self, value: str) -> None:
        self._last_name = value

    @property
    def age(self) -> int:
        return self._age

    @age.setter
    def age(self, value: int) -> None:
        if value < 0:
            raise ValueError("Age must be positive")
        self._age = value

    @property
    def gender(self) -> Gender:
        return self._gender

    @gender.setter
    def gender(self, value: Gender) -> None:
        self._gender = value

    def save_to_csv(self) -> int:
        """
        Save the user to CSV and return the assigned user_id.
        """
        
        csv_path = Path(config.CSV_USERS)  
        columns = ["user_id", "first_name", "last_name", "age", "gender"]

        # Load existing CSV or create empty DataFrame
        if csv_path.exists():
            df = pd.read_csv(csv_path, sep=";")
        else:
            df = pd.DataFrame(columns=columns)

        # Determine next user_id
        if "user_id" in df.columns and not df.empty:
            self._user_id = int(df["user_id"].max() + 1)
        else:
            self._user_id = 1

        # Create a dict of user data
        user_dict = {
            "user_id": self._user_id,
            "first_name": self._first_name,
            "last_name": self._last_name,
            "age": self._age,
            "gender": self._gender.value  
        }

        # Append to DataFrame and save
        df = pd.concat([df, pd.DataFrame([user_dict])], ignore_index=True)
        df.to_csv(csv_path, sep=";", index=False)

        return self._user_id

