import argparse
import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from call_function import available_functions, call_function
from prompts import system_prompt

parser = argparse.ArgumentParser(description="Chatbot")
parser.add_argument("user_prompt", type=str, help="User prompt")
parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
args = parser.parse_args()

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

if api_key is None:
    raise RuntimeError("GEMINI_API_KEY environment variable not found. Please add it to your .env file.")

client = genai.Client(api_key=api_key)

messages: list[types.Content] = [
    types.Content(role="user", parts=[types.Part(text=args.user_prompt)])
]

for i in range(20):
    for attempt in range(8):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=messages,
                config=types.GenerateContentConfig(
                    tools=[available_functions],
                    system_instruction=system_prompt,
                    temperature=0,
                ),
            )
            break
        except Exception:
            if attempt == 7:
                raise
            time.sleep(10)

    if args.verbose and response.usage_metadata is not None:
        print(f"User prompt: {args.user_prompt}")
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Response tokens: {response.usage_metadata.candidates_token_count}")

    if response.candidates:
        for candidate in response.candidates:
            messages.append(candidate.content)

    if response.function_calls:
        function_responses = []

        for function_call in response.function_calls:
            function_call_result = call_function(function_call, args.verbose)

            if not function_call_result.parts:
                raise Exception("Function call returned no parts.")

            function_response = function_call_result.parts[0].function_response
            if function_response is None:
                raise Exception("Function call returned no function response.")

            if function_response.response is None:
                raise Exception("Function call returned no response.")

            function_responses.append(function_call_result.parts[0])

            if args.verbose:
                print(f"-> {function_response.response}")

        messages.append(types.Content(role="user", parts=function_responses))
    else:
        print("Final response:")
        print(response.text)
        sys.exit(0)

print("Error: maximum iterations reached")
sys.exit(1)
