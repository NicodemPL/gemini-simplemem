import argparse
import os
import sys
from datetime import datetime

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Monkey patch config before importing main
import config

from main import SimpleMemSystem

def main():
    parser = argparse.ArgumentParser(description="SimpleMem CLI for Gemini")
    parser.add_argument("--api-key", help="Gemini API Key")
    parser.add_argument("action", choices=["add", "query", "clear"], help="Action to perform")
    parser.add_argument("text", nargs="?", help="Text to add or query")
    parser.add_argument("--date", help="Date for the memory (ISO format)", default=None)
    
    args = parser.parse_args()

    # Set API Key
    if args.api_key:
        config.OPENAI_API_KEY = args.api_key
        os.environ["GEMINI_API_KEY"] = args.api_key
        os.environ["GOOGLE_API_KEY"] = args.api_key
        os.environ["OPENAI_API_KEY"] = args.api_key 
    
    # Ensure environment variables are set for litellm
    if config.OPENAI_API_KEY:
        os.environ["GEMINI_API_KEY"] = config.OPENAI_API_KEY
        os.environ["GOOGLE_API_KEY"] = config.OPENAI_API_KEY
        os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

    # Initialize System
    clear_db = (args.action == "clear")
    
    try:
        system = SimpleMemSystem(clear_db=clear_db)
    except Exception as e:
        print(f"Error initializing system: {e}")
        return

    if args.action == "add":
        if not args.text:
            print("Error: Text required for 'add' action.")
            return
        
        timestamp = args.date or datetime.now().isoformat()
        print(f"Adding memory: '{args.text}' at {timestamp}")
        system.add_dialogue("User", args.text, timestamp)
        system.finalize()
        print("✅ Memory added successfully.")

    elif args.action == "query":
        if not args.text:
            print("Error: Text required for 'query' action.")
            return
            
        print(f"🔎 Querying: '{args.text}'")
        try:
            answer = system.ask(args.text)
            print("\n=== Answer ===")
            print(answer)
            print("==============")
        except Exception as e:
            print(f"Error during query: {e}")

    elif args.action == "clear":
        print("✅ Memory cleared.")

if __name__ == "__main__":
    main()
