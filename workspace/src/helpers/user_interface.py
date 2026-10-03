from enum import Enum
import sys
from datetime import datetime

class UserInterface:
    """
    Helper class which provides methods for interacting with the user via console.
    """

    def __init__(self, prefix="[INFO]", 
                 error_prefix="[ERROR]", 
                 input_prefix="[INPUT]", 
                 confirmation_prefix="[CONFIRMATION]", 
                 selection_prefix = "[SELECTION]"):
        
        self.prefix = prefix
        self.error_prefix = error_prefix
        self.input_prefix = input_prefix
        self.confirmation_prefix = confirmation_prefix
        self.selection_prefix = selection_prefix

    def print_info(self, message: str) -> None:
        """Print info log"""
        print(f"{datetime.now()} | {self.prefix} | {message}")

    def print_error(self, message: str) -> None:
        """Print an error log"""
        print(f"{datetime.now()} | {self.error_prefix} {message}", file=sys.stderr)

    def get_input(self, prompt: str,  cast_type=None):
        """
        Ask the user for input, optionally converting it to a specific type.

        - If cast_type is provided (e.g., int, float), it will try to convert the input.
        - If conversion fails, the user will be asked again.

        Returns:
            User input (converted if cast_type provided).
        """
        while True:
            try:
                value = input(f"{self.input_prefix} {prompt}: ").strip()

                if not value:
                    self.print_error("Input cannot be empty. Please try again.")
                    continue

                if cast_type:
                    try:
                        return cast_type(value)
                    except ValueError:
                        self.print_error(f"Invalid input. Please enter a valid {cast_type.__name__}.")
                else:
                    return value
                
            except KeyboardInterrupt:
                print("\nUser interrupted. Exiting...")
                sys.exit(0)

    def get_confirmation(self, prompt: str) -> bool:
        """
        Ask the user for confirmation (y/n).

        Returns:
            true -> if confirmed
            false -> if rejected
        """
        while True:
            try:
                value = input(f"{self.confirmation_prefix} {prompt} (y/n): ").strip()

                if not value:
                    self.print_error("Input cannot be empty. Please try again.")
                    continue

                if value == "y":
                    return True
                elif value =="n":
                    return False
                else:
                    self.print_error(f"Invalid input. Please enter y/n values.")

            except KeyboardInterrupt:
                print("\nUser interrupted. Exiting...")
                sys.exit(0)

    def get_selection(self, prompt: str,  options: list[str]) -> str:
        """
        Allows user to select string value from the list.

        """

        print(f"{self.selection_prefix} {prompt}")

        for i, option in enumerate(options, start=1):
            print(f"{i}. {option}")

        while True:
            try:
                choice = int(input("Enter number: "))
                if 1 <= choice <= len(options):
                    return options[choice - 1]
                else:
                    self.print_error(f"Invalid input. Please enter valid value from the list.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nUser interrupted. Exiting...")
                sys.exit(0)