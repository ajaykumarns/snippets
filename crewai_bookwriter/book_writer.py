# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "crewai",
#     "crewai-tools",
#     "pydantic",
#     "openai",
#     "httpx",
# ]
# ///


# Inspired from Avi Chawla's tweets: https://x.com/_avichawla/status/1900434449797701942, working example
import asyncio
import os
from crewai import Agent, Crew, Process, Task, LLM
from pydantic import BaseModel, Field
from crewai.flow import Flow, listen, start
from crewai.tools import BaseTool
from typing import Type
import requests

# llm = ChatOllama(model="ollama/gemma3:4b")
AGENT_VERBOSE = False
CREW_VERBOSE = True
llm = LLM(
    # model="ollama/deepseek-r1:7b"
    # model="ollama/qwen2.5:3b"
    model="ollama/qwen2.5:7b"
    # model="ollama/gemma3:4b"
    # model="ollama/llama3.2"
)

# BrightData Web Search Tool


class BrightDataWebSearchToolInput(BaseModel):
    title: str = Field(..., description="Topic of the book for which to write about.")


class BrightDataWebSearchTool(BaseTool):
    name: str = "BrightData Web Search"
    description: str = "Tool to search Google and retrieve the results."
    args_schema: Type[BaseModel] = BrightDataWebSearchToolInput

    def _run(self, title: str) -> str:
        host = os.getenv("BD_HOST", "brd.superproxy.io")
        port = os.environ["BD_PORT"]
        username = os.environ["BD_USERNAME"]
        password = os.environ["BD_PASSWORD"]
        proxies = {
            "http": f"http://{username}:{password}@{host}:{port}",
            "https": f"http://{username}:{password}@{host}:{port}",
        }
        url = f"https://www.google.com/search?q={title}&brd_json=1&num=500"
        response = requests.get(url, proxies=proxies, verify=False)
        return response.json()["organic"]


# Define pydantic models


class Outline(BaseModel):
    total_chapters: int
    titles: list[str]


class Chapter(BaseModel):
    title: str
    content: str


class BookState(BaseModel):
    topic: str = (
        """From waste to gold, easy composting techniques for the home gardener."""
    )
    total_chapters: int = 0
    titles: list[str] = []
    chapters: list[Chapter] = []


# Agents and task definitions

research_agent = Agent(
    role="Book Researcher: [{topic}]",
    goal="Research the topic [{topic}] and collect information about it.",
    backstory="You are an expert at [{topic}] and have a deep understanding and insights about it.",
    tools=[BrightDataWebSearchTool()],
    llm=llm,
)

research_task = Task(
    description="Prepare insights and key points that are needed to create an outline for a book about [{topic}]. Make sure you find any interesting and relevant information given the current year is 2025.",
    expected_output="Insights and key points about [{topic}].",
    agent=research_agent,
)

outline_agent = Agent(
    role="Book Outline Writer: [{topic}]",
    goal="Create an outline for a book about [{topic}].",
    backstory="You are an expert on [{topic}] and at creating outlines for books.",
    llm=llm,
)

outline_task = Task(
    description="Create an outline for a book about [{topic}]",
    expected_output="Total number of chapters and titles for each chapter.",
    agent=outline_agent,
    output_pydantic=Outline,
)

chapter_research_agent = Agent(
    role="Chapter Researcher: [{title}]",
    goal="Research the topic [{title}] and collect information about it.",
    backstory="You are an expert at [{title}] and have a deep understanding and insights about it.",
    tools=[BrightDataWebSearchTool()],
    llm=llm,
)

chapter_research_task = Task(
    description="Research and prepare insights about [{title}] for the book [{topic}].",
    expected_output="Insights and key points about [{title}].",
    agent=chapter_research_agent,
)

chapter_writer_agent = Agent(
    role="Senior Writer: [{title}]",
    goal="Write a chapter about [{title}] for the book [{topic}].",
    backstory="You are an expert at writing about [{title}] and have a deep understanding and insights about it.",
    tools=[],
    llm=llm,
)

chapter_writer_task = Task(
    description="Write a chapter about [{title}] for the book [{topic}].",
    expected_output="A chapter about [{title}] for the book [{topic}].",
    agent=chapter_writer_agent,
    output_pydantic=Chapter,
)

OutlineCrew = Crew(
    agents=[research_agent, outline_agent],
    tasks=[research_task, outline_task],
    process=Process.sequential,
    llm=llm,
    verbose=CREW_VERBOSE,
)

ChapterWriterCrew = Crew(
    agents=[chapter_research_agent, chapter_writer_agent],
    tasks=[chapter_research_task, chapter_writer_task],
    process=Process.sequential,
    llm=llm,
    verbose=CREW_VERBOSE,
)

# Flow definition to generate book


class BookFlow(Flow[BookState]):
    def __init__(self, book_dir: str):
        super().__init__()
        self.book_dir = book_dir

    @start()
    def generate_outline(self):
        outline = OutlineCrew.kickoff(inputs={"topic": self.state.topic})
        self.state.total_chapters = outline.pydantic.total_chapters
        self.state.titles = outline.pydantic.titles

    @listen(generate_outline)
    async def generate_chapters(self):
        tasks = []

        async def write_single_chapter(chapter_title: str):
            try:
                print(
                    f"Kicking off chapter writer crew for: title={chapter_title} and topic={self.state.topic}"
                )
                result = ChapterWriterCrew.kickoff(
                    inputs={
                        "title": chapter_title,
                        "topic": self.state.topic,
                        "chapters": self.state.titles,
                    }
                )

                os.makedirs(f"{self.book_dir}/chapters", exist_ok=True)
                with open(f"{self.book_dir}/chapters/{chapter_title}.md", "w") as f:
                    f.write(f"# [{chapter_title}]\n")
                    f.write(result.pydantic.content)
                    f.write("\n")
                    print(f"Finished writing chapter {chapter_title}")
                return result.pydantic
            except Exception as e:
                print(f"Error generating chapter '{chapter_title}': {str(e)}")
                return e

        for chapter in self.state.titles:
            tasks.append(write_single_chapter(chapter))
        self.state.chapters = await asyncio.gather(*tasks, return_exceptions=True)

    @listen(generate_chapters)
    def save_chapters(self):
        print("Total number of chapters: ", len(self.state.chapters))
        with open(f"{self.book_dir}/book.md", "w") as f:
            for i, chapter in enumerate(self.state.chapters):
                if isinstance(chapter, Exception):
                    print(f"Writing error for chapter {i + 1}")
                    f.write(f"# ERROR: Failed Chapter {i + 1}\n")
                    f.write(f"Unable to generate chapter due to: {str(chapter)}\n\n")
                else:
                    print(f"Writing chapter {i + 1}: {chapter.title}")
                    f.write(f"# {chapter.title}\n")
                    f.write(f"{chapter.content}\n\n")
            print("Finished writing all chapters")

        # Count successful and failed chapters
        successful = sum(1 for c in self.state.chapters if not isinstance(c, Exception))
        failed = sum(1 for c in self.state.chapters if isinstance(c, Exception))

        print(f"Book saved with {successful} successful and {failed} failed chapters.")

        # Don't raise an exception here, just report the failures
        if failed > 0:
            print(
                f"Warning: {failed} chapters failed to generate. See book.md for details."
            )


if __name__ == "__main__":
    book_dir = f"{os.path.expanduser('~')}/books"
    os.makedirs(book_dir, exist_ok=True)
    flow = BookFlow(book_dir=book_dir)
    flow.kickoff()
