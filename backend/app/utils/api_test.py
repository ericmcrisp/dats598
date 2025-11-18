from openai import OpenAI
from app.core.config import settings


def test_openai():
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # print(settings.OPENAI_API_KEY)
    
    response = client.responses.create(
        model="gpt-5-nano",
        input="is the sky blue?",
        store=False
    )

    print(response.output_text)

if __name__ == "__main__":
    test_openai()
