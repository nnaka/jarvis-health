from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
import uvicorn
import sys
load_dotenv()

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")

class UserInput(BaseModel):
    query: str
    soldier_data: dict

# Logging configuration
logging_config = {
    "version": 1,
    "formatters": {
        "json": {
            "format": "%(asctime)s %(process)s %(levelname)s %(name)s %(module)s %(funcName)s %(lineno)s"
        }
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "json",
            "stream": sys.stderr,
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": [
            "console"
        ],
        "propagate": True
    }
}

@app.post("/optimize")
async def generate_optimization_plan(user_input: UserInput):
    # return {"plan": user_input.query}
    try:
        context = fetch_relevant_context(user_input.soldier_data)
        prompt = orchestrate_prompt(user_input.query, context)
        print(f"Prompt: {prompt}")
        raw_plan = call_llm_api(prompt)
        print(f"Raw plan: {raw_plan}")
        validated_plan = validate_output(raw_plan)
        print(f"Validated plan: {validated_plan}")
        enhanced_plan = enhance_plan_with_tools(validated_plan)
        print(f"Enhanced plan: {enhanced_plan}")
        return {"plan": enhanced_plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def fetch_relevant_context(soldier_data):
    # Implement context fetching logic
    
    pass

def orchestrate_prompt(query, context):
    # Construct a structured prompt that combines the user query and soldier context
    system_message = """You are a military training optimization assistant. Analyze the provided soldier data 
    and training query to generate specific, actionable recommendations. Focus on safety, effectiveness, 
    and standard military protocols."""
    
    # Format the context data into a readable string
    context_str = "No context provided"
    if context:
        context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
    
    user_message = f"""
    Soldier Data:
    {context_str}
    
    Training Query:
    {query}
    
    Please provide a detailed optimization plan that:
    1. Addresses the specific training query
    2. Takes into account the soldier's current data
    3. Includes specific, measurable goals
    4. Maintains safety standards
    5. Follows military protocols
    """
    
    # Return the formatted messages for the OpenAI API
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def call_llm_api(prompt):
    # Implement OpenAI API call
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        temperature=0.5,
        max_tokens=1000
    )
    return response.choices[0].message.content

import re

def validate_output(raw_plan):
    # Define validation rules
    max_exercise_duration = 120  # Maximum exercise duration in minutes
    min_rest_period = 6  # Minimum rest period in hours
    max_daily_calories = 4000  # Maximum recommended daily calorie intake
    forbidden_words = ["steroids", "drugs", "shortcuts"]

    # Initialize validation result
    validation_result = {
        "is_valid": True,
        "warnings": [],
        "modified_plan": raw_plan
    }

    # Check exercise duration
    exercise_durations = re.findall(r'(\d+)\s*minutes?', raw_plan, re.IGNORECASE)
    for duration in exercise_durations:
        if int(duration) > max_exercise_duration:
            validation_result["warnings"].append(f"Exercise duration of {duration} minutes exceeds the recommended maximum of {max_exercise_duration} minutes.")
            validation_result["is_valid"] = False

    # Check rest periods
    rest_periods = re.findall(r'rest\s*for\s*(\d+)\s*hours?', raw_plan, re.IGNORECASE)
    for period in rest_periods:
        if int(period) < min_rest_period:
            validation_result["warnings"].append(f"Rest period of {period} hours is below the recommended minimum of {min_rest_period} hours.")
            validation_result["is_valid"] = False

    # Check calorie recommendations
    calorie_recommendations = re.findall(r'(\d+)\s*calories', raw_plan, re.IGNORECASE)
    for calories in calorie_recommendations:
        if int(calories) > max_daily_calories:
            validation_result["warnings"].append(f"Recommended calorie intake of {calories} exceeds the maximum of {max_daily_calories} calories per day.")
            validation_result["is_valid"] = False

    # Check for forbidden words
    for word in forbidden_words:
        if word.lower() in raw_plan.lower():
            validation_result["warnings"].append(f"The plan contains the forbidden word: '{word}'.")
            validation_result["is_valid"] = False
            # Remove the forbidden word from the plan
            validation_result["modified_plan"] = re.sub(word, "[REDACTED]", validation_result["modified_plan"], flags=re.IGNORECASE)

    # Check for military-specific terminology
    military_terms = ["PT", "ruck march", "combat fitness"]
    if not any(term.lower() in raw_plan.lower() for term in military_terms):
        validation_result["warnings"].append("The plan lacks specific military fitness terminology.")

    return validation_result


def enhance_plan_with_tools(validated_plan):
    # Implement tool calling logic
    return validated_plan

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
