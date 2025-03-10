from utils.difficulty import Difficulty, AUDIENCES, WORD_COUNTS

from langchain.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama


def text_to_story(text: str, difficulty: Difficulty) -> str:
    if difficulty not in Difficulty:
        raise ValueError(f"Invalid difficulty: {difficulty}")

    audience = AUDIENCES[difficulty]
    min_words, max_words = WORD_COUNTS[difficulty]

    llm = ChatOllama(model="llama3")

    prompt = ChatPromptTemplate(
        [
            # sometimes the AI outputs some text that is not part of the story, e.g. "Here is a story about the sky: One sunny day, ...", so we need to tell the AI to only output the story"
            (
                "system",
                f"You are a helpful AI bot that generates story for {audience} from supplied text. The story should be more than {min_words} words and less than {max_words} words. Don't output any text that is not part of the story.",
            ),
            # Means the template will receive an optional list of messages under
            # the "chat_history" key
            ("placeholder", "{chat_history}"),
            # Equivalently:
            # MessagesPlaceholder(variable_name="chat_history", optional=True)
            ("human", "{message}"),
        ]
    )

    chat_history = []

    chain = prompt | llm

    story = chain.invoke({"message": text, "chat_history": chat_history})
    print("chain result: ", story)

    story_text = story.content
    # print('story: ', story_text)

    return story_text
