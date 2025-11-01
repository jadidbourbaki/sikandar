"""
Sikandar loop that keeps the sikandar model running and generates text based on user input.
"""
import subprocess


def generate_text(prompt: str) -> str:
    """
    Generate text based on a prompt using the sikandar model.
    """
    # system call for the make target sikandar
    result = subprocess.run(["make", "sikandar", f"PROMPT={prompt}"],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def sikandar_loop() -> None:
    """
    Loop that keeps the sikandar model running and generates text based on user input.
    """
    print("hello, i am sikandar!")
    print("i have been trained on the works of shakespeare.")
    print("you can type the start of a sentence and i will")
    print("continue it in the style of shakespeare.")
    print("note: to exit, type 'exit'.")
    while True:
        user_input = input("you: ")
        if user_input.lower() == "exit":
            print("goodbye!")
            break
        print(f"sikandar: {generate_text(user_input)}")


if __name__ == "__main__":
    sikandar_loop()
