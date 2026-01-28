import os
from core.chatbot_utils import ChatbotBase
from core.chatbot_utils import LocalPortSaverTransport

from openai import AzureOpenAI
import httpx
from dotenv import load_dotenv

class AzureGPT4o(ChatbotBase):
    """
        Azure GPT 4o chatbot.
    """

    def __init__(self, remote_tls_port=443):
        """
            Creates an instance.
        """

        # Call superclass
        super().__init__(remote_tls_port)

        # Load environment variables from .env file
        load_dotenv()

        # Validate environment variables
        if not os.getenv('AZURE_OPENAI_ENDPOINT'):
            raise ValueError('AZURE_OPENAI_ENDPOINT is not set in the environment variables.')
        if not os.getenv('AZURE_OPENAI_API_KEY'):
            raise ValueError('AZURE_OPENAI_API_KEY is not set in the environment variables.')

        # Create client that also saves the local port
        self._transport = LocalPortSaverTransport()
        self._client = AzureOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            api_version='2024-02-01',
            http_client=httpx.Client(transport=self._transport))

    def send_prompt(self, prompt, temperature):
        """
            Sends a prompt. Pulls data back as fast as possible (asynchronously) but waits.
            Returns a tuple of (response, local_port) - if local port cannot be determined return (response, None).
        """

        # Send prompt
        response = []
        stream = self._client.chat.completions.create(
            extra_body={},
            model='gpt-4o-adhoc',
            messages=[ { 'role': 'user', 'content': prompt } ],
            stream=True,
            temperature=temperature
        )
        for chunk in stream:
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                response.append(chunk.choices[0].delta.content)

        # Return response
        return (response, self._transport.get_local_port())

    def get_temperature(self):
        """
            Gets the temperature of the model.
        """

        # For now we just return the default of 1.0
        return 1.0

    def get_common_name(self):
        """
            Gets the common name of the model.
        """

        # Return common name
        return 'gpt-4o (Azure)'

    def match_tls_server_name(self, server_name):
        """
            Matches the TLS server name.
        """
        
        # Match
        return 'openai.azure.com' in server_name

