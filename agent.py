import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.adk.agents import LlmAgent, SequentialAgent

model = genai.GenerativeModel('gemini-2.0-flash')

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Configure API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env file")
genai.configure(api_key=api_key)

# Tool functions that call Gemini
def generate_ideas(topic: str) -> str:
    """Ask Gemini to generate ideas for a given topic."""
    try:
        response = model.generate_content(
            f"Brainstorm 4-6 creative Instagram post ideas for the topic: {topic}"
        )
        return response.text 
    except Exception as e:
        raise RuntimeError(f"Error in generate_ideas: {e}")

def write_content(ideas: str) -> str:
    """Ask Gemini to expand an outline into a -300-word draft."""
    try:
        response = model.generate_content(
            f"Expand the following outline into a cohesive -300-word Instagram post: {ideas}"
        )
        return response.text  # Correct attribute
    except Exception as e:
        raise RuntimeError(f"Error in write_content: {e}")

def format_draft(draft: str) -> str:
    """Ask Gemini to format the draft as clean Markdown."""
    try:
        response = model.generate_content(
            f"Format this draft as clean Markdown with headings, sub-headings, and bullet lists: {draft}"
        )
        return response.text  # Correct attribute
    except Exception as e:
        raise RuntimeError(f"Error in format_draft: {e}")

# Define three LLM agents
topic_agent = LlmAgent(
    name="IdeaAgent",
    model="gemini-2.0-flash",
    description="Brainstorms Instagram post ideas.",
    instruction="Call generate_ideas(topic) with the exact topic string you receive and return only the ideas.",
    tools=[generate_ideas],
    output_key="ideas",
)

draft_agent = LlmAgent(
    name="WriteAgent",
    model="gemini-2.0-flash",
    description="Write an Instagram post draft from ideas.",
    instruction="Call write_content(ideas) where 'ideas' is the output of the prior step, and return only the draft text.",
    tools=[write_content],
    output_key="draft",
)

format_agent = LlmAgent(
    name="FormatterAgent",
    model="gemini-2.0-flash",
    description="Formats the draft into Markdown.",
    instruction="Call format_draft(draft) where 'draft' is the previous output, and return only the final Markdown.",
    tools=[format_draft],
    output_key="formatted",
)

# Put everything in the pipeline
root_agent = SequentialAgent(
    name="ContentAssistant",
    sub_agents=[topic_agent, draft_agent, format_agent],
    description="Takes a user topic -> generates ideas -> writes draft -> formats as Markdown",
)